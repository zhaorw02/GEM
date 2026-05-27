import os
import math
import json
import random
import fnmatch
from typing import Dict, Sequence

import h5py
import yaml
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from train.image_corrupt import image_corrupt


class H5VLADataset(Dataset):
    def __init__(
        self,
        config,
        hdf5_dir: str,
        tokenizer,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
    ):
        super().__init__()

        self.image_aspect_ratio = config["dataset"]["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        
        self.camera_names = config['dataset']['camera_names']
        self.epsd_len_thresh_low = config['dataset']['epsd_len_thresh_low']
        # self.epsd_len_thresh_high = config['dataset']['epsd_len_thresh_high'] # TODO

        # Load the config
        self.action_chunk_size = config['common']['action_chunk_size']
        self.img_history_size = config['common']['img_history_size']
        
        self.grip_or_hand_latency_frames = config['common'].get('grip_or_hand_latency_frames', 10)
        self.arm_latency_frames = config['common'].get('arm_latency_frames', 3)

        # hdf5 dataset
        self.hdf5_dir = hdf5_dir
        # Create data indexes
        index_path = os.path.join(self.hdf5_dir, "index.json")
        if False and os.path.exists(index_path):    # TODO: multiprocessing: one will create incompletely and the other read...
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            self.h5_file_paths = index_data['h5_file_paths']
            self.valid_episode_lens = index_data['valid_episode_lens']
            self.first_step_ids = index_data['first_step_ids']
            self.num_steps = index_data['num_steps']
            self.file_path_to_idx = index_data['file_path_to_idx']
            self.total_steps = index_data['total_steps']
            self.index_to_h5_file_path = index_data['index_to_h5_file_path']
            self.index_to_h5_file_path = {
                int(k): v for k, v in index_data['index_to_h5_file_path'].items()
            }
        else:
            self.h5_file_paths = []
            for root, _, files in os.walk(self.hdf5_dir):
                for filename in fnmatch.filter(files, '*.hdf5'):
                    file_path = os.path.join(root, filename)
                    self.h5_file_paths.append(file_path)

            self.valid_episode_lens = []
            self.first_step_ids = []
            self.num_steps = []
            self.file_path_to_idx = {}
            self.total_steps = 0
            self.index_to_h5_file_path: dict[int, tuple[str, int]] = {} # global_index: (h5_file_path, step_id)
            for i, h5_file_path in tqdm(enumerate(self.h5_file_paths), desc="Indexing HDF5 files..."):
                h5_valid, first_step_id, num_step = self.check_h5(h5_file_path)
                valid_episode_len = num_step - first_step_id if h5_valid else 0
                self.valid_episode_lens.append(valid_episode_len)
                self.first_step_ids.append(first_step_id)
                self.num_steps.append(num_step)
                self.file_path_to_idx[h5_file_path] = i
                for j in range(valid_episode_len):
                    self.index_to_h5_file_path[self.total_steps + j] = (h5_file_path, first_step_id + j)
                self.total_steps += valid_episode_len
            # Save the index
            index_data = {
                'h5_file_paths': self.h5_file_paths,
                'valid_episode_lens': self.valid_episode_lens,
                'first_step_ids': self.first_step_ids,
                'num_steps': self.num_steps,
                'file_path_to_idx': self.file_path_to_idx,
                'total_steps': self.total_steps,
                'index_to_h5_file_path': self.index_to_h5_file_path
            }
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=4)
        
        # Load the normalization stat from `action_stats.json`
        action_stats_path = os.path.join(self.hdf5_dir, "action_stats.json")
        if os.path.exists(action_stats_path):
            with open(action_stats_path, 'r') as f:
                action_stats = json.load(f)
            self.action_min = np.array(action_stats['action_min'])
            self.action_max = np.array(action_stats['action_max'])  # (a_dim,)
        else:
            self.action_min, self.action_max = None, None

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug

    def check_h5(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            frist_step_id (int): the index of the first step that is not still,
                i.e., the first step whose qpos exceeds the threshold.
            num_steps (int): the total number of steps in the episode.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['state'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < self.epsd_len_thresh_low:
                return False, None, None

            return True, 0, num_steps

            # TODO: move this part to data preprocessing
            # [Optional] We skip the first few still steps
            # EPS = 1e-2
            # # Get the idx of the first qpos whose delta exceeds the threshold
            # qpos_delta = np.abs(qpos - qpos[0:1])
            # indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            # if len(indices) > 0:
            #     first_idx = indices[0]
            #     valid_episode_len = num_steps - (first_idx - 1)

            #     return True, first_idx - 1, num_steps

            # return False, None, None

    def parse_hdf5_file(self, file_path, step_id: int):
        """
        Src H5 Structure:
            action (Dataset)
                Shape: (N, a_dim), Dtype: float32
            observations (Group)
                images (Group)
                    <cam_1> (Dataset)
                    Shape: (N,), Dtype: |S40718
                    <cam_2> (Dataset)
                    Shape: (N,), Dtype: |S33848
                    ...
                state (Dataset)
                    Shape: (N, s_dim), Dtype: float32

        `state` means the proprioceptive state of the robot.
        Camera names are defined in the `camera_names` entry of the config file.
        Images could be stored in bytes (jpeg compression).

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, state_dim).
                    "state_std": ndarray,       # std(state[:]), (state_dim,).
                    "action_norm": ndarray,     # norm(action[:]), (action_dim,).
                    "actions": ndarray,         # action[t:t+action_chunk_size], (action_chunk_size, action_dim).
                    "<cam_1>": ndarray,         # external camera image, (img_history_size, H, W, 3)
                    ...
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            state = f['observations']['state'][:] # (N, s_dim)
            if 'action_chunk' in f.keys():
                action_chunk = f['action_chunk'][:]
            else:
                action_chunk = None
                action = f['action'][:] # (N, a_dim)
            num_steps = state.shape[0]

            # first_step_id = self.first_step_ids[self.file_path_to_idx[file_path]]

            # Load the instruction
            dir_path = os.path.dirname(file_path)
            with open(os.path.join(dir_path, 'instructions.json'), 'r') as f_instr:
                instruction_list = json.load(f_instr)
            chosen_idx = random.randint(0, len(instruction_list) - 1)
            instruction = instruction_list[chosen_idx]

            # Assemble the meta
            meta = {
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # TODO: move latency matching to data preprocessing
            # Action should already consider latency
            # target_qpos = f['action'][step_id: step_id + self.action_chunk_size] # for real, diff by 1 tick
            # for robotwin sim, qpos is the same as target_qpos
            # action: [T + 1, T + 1 + action_chunk_size)
            # consider latency:
            # arm action: [T + 1 + latency_frames, T + 1 + latency_frames + action_chunk_size)
            # target_qpos: (action_chunk_size, q_dim)
            # target_qpos = np.zeros((self.action_chunk_size, state.shape[1]), dtype=np.float32)
            # def fill_target_qpos_consider_latency(target_qpos: np.ndarray, step_id: int, num_steps: int, slice_x: slice, latency_frames: int, action_chunk_size: int):
            #     if step_id + 1 + latency_frames + action_chunk_size > num_steps:
            #         if step_id + 1 + latency_frames >= num_steps:
            #             target_qpos[:, slice_x] = state[-1:, slice_x].repeat(action_chunk_size, axis=0) # (action_chunk_size, slice_dim)
            #         else:
            #             target_qpos[:num_steps - (step_id + 1 + latency_frames), slice_x] = state[step_id + 1 + latency_frames:, slice_x] # (num_steps - (T + 1 + latency_frames), slice_dim)
            #             target_qpos[num_steps - (step_id + 1 + latency_frames):, slice_x] = np.tile(
            #                 state[-1:, slice_x], 
            #                 (action_chunk_size - (num_steps - (step_id + 1 + latency_frames)), 1)
            #             ) # (action_chunk_size - (num_steps - (T + 1 + latency_frames)), slice_dim)
            #     else:
            #         target_qpos[:, slice_x] = state[step_id + 1 + latency_frames: step_id + 1 + latency_frames + action_chunk_size, slice_x] # (action_chunk_size, slice_dim)
            
            # if self.eef_type == "grip":
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(0, 7), self.arm_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(7, 8), self.grip_or_hand_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(8, 15), self.arm_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(15, 16), self.grip_or_hand_latency_frames, self.action_chunk_size)
            # elif self.eef_type == "hand6dof":
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(0, 7), self.arm_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(7, 13), self.grip_or_hand_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(13, 20), self.arm_latency_frames, self.action_chunk_size)
            #     fill_target_qpos_consider_latency(target_qpos, step_id, num_steps, slice(20, 26), self.grip_or_hand_latency_frames, self.action_chunk_size)
            # else:
            #     raise NotImplementedError(f"Unsupported end-effector type: {self.eef_type}. ")

            # Parse the state and action
            state = state[step_id: step_id + 1] # (1, s_dim)
            state_std = np.std(state, axis=0) # (s_dim,)

            if action_chunk is None:
                action_norm = np.sqrt(np.mean(action ** 2, axis=0)) # float
                action_chunk = action[step_id: step_id + self.action_chunk_size] # (action_chunk_size, a_dim)
                if action_chunk.shape[0] < self.action_chunk_size:
                    # If the action chunk is shorter than the action chunk size,
                    # we pad it with the last action
                    action_chunk = np.concatenate([
                        action_chunk, 
                        np.tile(action_chunk[-1:], (self.action_chunk_size - action_chunk.shape[0], 1))
                    ], axis=0)
            else:
                action_norm = np.sqrt(np.mean(
                    action_chunk.reshape(-1, action_chunk.shape[-1]) ** 2, axis=0)) # float
                action_chunk = action_chunk[step_id]
                

            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.img_history_size+1, 0), step_id+1):
                    img = f['observations']['images'][key][i]
                    # If img is 1D, decode it
                    if img.ndim < 3:
                        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.img_history_size:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.img_history_size-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs

            # cam_high = parse_img('cam_high')
            # valid_len = min(step_id - first_step_id + 1, self.img_history_size)
            # cam_high_mask = np.array(
            #     [False] * (self.img_history_size - valid_len) + [True] * valid_len
            # )
            
            rt_dict = {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "action_norm": action_norm,
                "actions": action_chunk
            }
            
            for camera_name in self.camera_names:
                rt_dict[camera_name] = parse_img(camera_name)
            
            return rt_dict

    # @staticmethod
    # def pairwise(iterable):
    #     a = iter(iterable)
    #     return zip(a, a)

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, index):
        # Iterate all the episodes one by one
        h5_file_path, step_id = self.index_to_h5_file_path[index]
        first_step_id = self.first_step_ids[self.file_path_to_idx[h5_file_path]]
        valid_episode_len = self.valid_episode_lens[self.file_path_to_idx[h5_file_path]]
        h5_sample = self.parse_hdf5_file(h5_file_path, step_id)

        content = h5_sample['meta']
        states = h5_sample['state']
        actions = h5_sample['actions']
        state_std = h5_sample['state_std']
        action_norm = h5_sample['action_norm']
        images = [h5_sample[camera_name] for camera_name in self.camera_names]

        data_dict = {}

        if self.state_noise_snr is not None:
            states += np.random.normal(
                0.0, state_std / np.sqrt(10 ** (self.state_noise_snr / 10)),
                states.shape)

        data_dict["states"] = states
        data_dict["actions"] = actions

        # Norm for the episode that the step belongs to
        # Used to calculate relative error
        data_dict["action_norm"] = action_norm

        # Min-max normalization if available
        if self.action_min is not None and self.action_max is not None:
            data_dict["actions"] = (data_dict["actions"] - self.action_min) / \
                (np.maximum(self.action_max - self.action_min, 1e-3)) * 2 - 1
            # TODO: it is buggy, should normalize before calculating norm
            data_dict["action_norm"] = (data_dict["action_norm"] - self.action_min) / \
                (np.maximum(self.action_max - self.action_min, 1e-3)) * 2 - 1

        mask_probs = [self.cond_mask_prob] * self.num_cameras
        if self.cam_ext_mask_prob >= 0.0:
            mask_probs[0] = self.cam_ext_mask_prob
        rearranged_images = []
        for i in range(self.img_history_size):
            for j in range(self.num_cameras):
                image = images[j][i]
                if (math.prod(image.shape) > 0) and \
                    (random.random() > mask_probs[j]):
                    rearranged_images.append((image, True))
                else:
                    rearranged_images.append((None, False))

        pil_images = []
        for image, valid in rearranged_images:
            if not valid or image is None:
                continue
            image = Image.fromarray(image)

            if self.auto_adjust_image_brightness:
                average_brightness = np.array(image).mean() / 255.0
                if average_brightness <= 0.36:
                    average_brightness = max(1e-2, average_brightness)
                    factor = min(2.0, 0.36 / average_brightness)
                    image = transforms.ColorJitter(brightness=(factor, factor))(image)

            # Only apply image augmentation to 50% of the images
            if self.image_aug and (random.random() > 0.5):
                aug_type = random.choice([
                    "corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)

            pil_images.append(image)

        data_dict["pil_images"] = pil_images

        instruction = content["instruction"] \
            if random.random() > self.cond_mask_prob else ""
        data_dict["instruction"] = instruction

        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
                # data_dict[k] = torch.from_numpy(v)

        data_dict["h5_file_path"] = h5_file_path
        data_dict["step_id"] = step_id
        data_dict["valid_episode_len"] = valid_episode_len
        data_dict["first_step_id"] = first_step_id

        return data_dict

    def _collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],

            "h5_file_paths": [],
            "step_ids": [],
            "valid_episode_lens": [],
            "first_step_ids": [],
        }

        texts = []
        all_images = []

        for instance in instances:
            batch["h5_file_paths"].append(instance["h5_file_path"])
            batch["step_ids"].append(instance["step_id"])
            batch["valid_episode_lens"].append(instance["valid_episode_len"])
            batch["first_step_ids"].append(instance["first_step_id"])

            # Convert all the numpy arrays to tensor
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # Build Qwen3VL conversation format
            content = []
            for img in instance["pil_images"]:
                content.append({"type": "image", "image": img})
                all_images.append(img)
            content.append({"type": "text", "text": instance["instruction"]})

            messages = [{"role": "user", "content": content}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # Create VLM inputs using the Qwen3VL processor
        vlm_inputs = self.tokenizer(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )
        batch["vision_language_model_inputs"] = vlm_inputs

        return batch

