import json
import random
import logging
import re
import time
import itertools
import tqdm
from dataclasses import dataclass, field

# Suppress verbose logs from qwen_vl_utils
logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)
logging.getLogger("qwen_vl_utils.vision_process").setLevel(logging.ERROR)
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import transformers
# from qwen_vl_utils import process_vision_info
from transformers.video_utils import VideoMetadata

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)



from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

from .utils import get_depth_path, load_depth_color_image
from .vision_process import process_vision_info


from PIL import Image

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

from torchvision.transforms import v2

class AdaptiveResize32:
    """
    将图像长边缩放到不超过 max_size，且宽高自动对齐为 32 的倍数。
    """
    def __init__(self, max_size=256):
        self.max_size = max_size

    def __call__(self, img):
        w, h = (img.width, img.height) if hasattr(img, 'width') else (img.shape[-1], img.shape[-2])
        scale = self.max_size / max(h, w)
    
        new_h = (int(h * scale) // 32) * 32
        new_w = (int(w * scale) // 32) * 32
        
        new_h, new_w = max(new_h, 32), max(new_w, 32)
        
        return F.resize(img, [new_h, new_w], antialias=True)
    
    
## target transform for sana
target_transform = v2.Compose(
    [
        AdaptiveResize32(max_size=448),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str | List[str]) -> str:
    if isinstance(files, list):
        return [f"{(base / file).resolve()}" for file in files]
    elif isinstance(files, str):
        return f"{(base / files).resolve()}"



def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    processor_kwargs = {}

    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [{"type": "image", "image": _make_abs_paths(base_path, img)} for img in images]
    video_pool = [{"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)
            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages




def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)
    
    # Inject default min_pixels/max_pixels if not present
    image_processor = getattr(processor, "image_processor", None)
    video_processor = getattr(processor, "video_processor", None)

    img_defaults = {}
    if image_processor:
        img_defaults = {
            "min_pixels": getattr(image_processor, "min_pixels", None),
            "max_pixels": getattr(image_processor, "max_pixels", None),
        }

    vid_defaults = {}
    if video_processor:
        vid_size = getattr(video_processor, "size", {})
        vid_defaults = {
            "min_pixels": getattr(image_processor, "min_pixels", None),
            "total_pixels": getattr(video_processor, "max_pixels", None) or vid_size.get("longest_edge"),
            "min_frames": getattr(video_processor, "min_frames", None),
            "max_frames": getattr(video_processor, "max_frames", None),
            "fps": getattr(video_processor, "fps", None),
        }

    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                for k, v in img_defaults.items():
                    if k not in content and v is not None:
                        content[k] = v
            elif content["type"] == "video":
                for k, v in vid_defaults.items():
                    if k not in content and v is not None:
                        content[k] = v
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
 
    
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )


    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        
        image_inputs = videos[0]
        video_images = True
        
        n_images_per_video = 32 # treat a video as 32 separate images, and replace one video token with 32 image tokens
        text = text.replace("<|video_pad|>", "<|image_pad|>"*n_images_per_video)
    else:
        video_metadatas = None
        image_inputs = images
        video_images = False
    
    full_result = processor(
        text=text,
        images=image_inputs,
        video_metadata=None,
        video_images=video_images,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
   
    
    input_ids = full_result["input_ids"]
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1
  
    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    
    full_result["images"] = images
    full_result["videos"] = videos
   
    return full_result, video_metadatas


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
        


        dataset = data_args.dataset_use.split(",")
        list_data_dict_all = []

        for meta_dataset in dataset:
            if meta_dataset.endswith(".json"):
                rank0_print(f"Loading datasets: {meta_dataset}")
                list_data_dict = json.load(open(meta_dataset, "r"))
            else:
                dataset_list = data_list([meta_dataset])
                rank0_print(f"Loading datasets: {dataset_list}")

                list_data_dict = []

                for data in tqdm.tqdm(dataset_list, desc="Loading datasets"):
                    file_format = data["annotation_path"].split(".")[-1]
                    if file_format == "jsonl":
                        annotations = read_jsonl(data["annotation_path"])
                    else:
                        annotations = json.load(open(data["annotation_path"], "r"))

                    sampling_rate = data.get("sampling_rate", 1.0)
                    if sampling_rate < 1.0:
                        annotations = random.sample(
                            annotations, int(len(annotations) * sampling_rate)
                        )
                        rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
                    else:
                        rank0_print(f"dataset name: {data}")

                    for ann in annotations:
                        if isinstance(ann, list):
                            for sub_ann in ann:
                                sub_ann["data_path"] = data["data_path"]
                                sub_ann["data_source"] = data["data_source"]
                        else:
                            ann["data_path"] = data["data_path"]
                            ann["data_source"] = data["data_source"]
                    list_data_dict += annotations
            list_data_dict_all += list_data_dict

        print(f"Total training samples: {len(list_data_dict_all)}")

        random.shuffle(list_data_dict_all)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict_all

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item
      
            
     

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 10
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_final_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def process_target_image(self, image):
        image = target_transform(image)
        return image
    
    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        
        data_dict, video_metadatas = preprocess_qwen_visual(
            sources,
            self.processor,
        )
        
        depth_image_paths = get_depth_path(sources[0], video_metadatas)
        
        if video_metadatas is not None:
            rgbs = data_dict["videos"][0]
            t, c, height, width = rgbs.shape 
        else:  
            rgbs = data_dict["images"]
            (width, height) = rgbs[0].size

      
        depth_images = [load_depth_color_image(depth_path, target_size=(width, height)) for depth_path in depth_image_paths]

        depth_images = [self.process_target_image(depth_image) for depth_image in depth_images]
        depth_images = torch.stack(depth_images, dim=0)
        data_dict["input_images"] = depth_images

            
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        

        
        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
        return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
     
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        
      
        max_len = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        position_ids = position_ids[:, :, :max_len]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids 
        )

       
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None
        
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw

    
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        
       
       
        input_images = list(
            instance["input_images"]
            for instance in instances
            if "input_images" in instance
        )
        if len(input_images) != 0:
            input_concat_images = torch.cat([input_image for input_image in input_images], dim=0)
        else:
            input_concat_images = None
        batch["input_images"] = input_concat_images
      
        

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        if input_ids.size(1) > self.tokenizer.model_max_length:
            print(f'Warning: input_ids length {input_ids.size(1)} exceeds model max length {self.tokenizer.model_max_length}. Truncating.')
            input_ids = input_ids[:, : self.tokenizer.model_max_length]
            labels = labels[:, : self.tokenizer.model_max_length]
            position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

       
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None
        
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw


        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

       
        input_images = list(
            instance["input_images"]
            for instance in instances
            if "input_images" in instance
        )
        if len(input_images) != 0:
            input_concat_images = torch.cat([input_image for input_image in input_images], dim=0)
        else:
            input_concat_images = None
        batch["input_images"] = input_concat_images
        

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

if __name__ == "__main__":
    pass
