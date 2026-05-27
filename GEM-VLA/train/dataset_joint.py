"""Extended dataset that loads both robot action data and depth images from HDF5.

Inherits from H5VLADataset and adds depth image loading for joint training.

Expected HDF5 structure for depth:
    observations/
        depth_images/
            <camera_name>/   (N,) uint8-encoded or (N, H, W) float/uint16
"""
import os
import random
from typing import Dict, Sequence

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train.dataset import H5VLADataset


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Convert a single-channel depth map to a 3-channel colorized image.

    Args:
        depth_map: (H, W) float array, values in [0, 1] (0=far, 1=near after inversion)

    Returns:
        (H, W, 3) uint8 array, colorized depth
    """
    depth_uint8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


class H5VLADepthDataset(H5VLADataset):
    """Dataset that returns both action data and depth images for joint training."""

    def __init__(
        self,
        *args,
        depth_camera_name: str = None,
        depth_image_size: int = 256,
        **kwargs,
    ):
        """
        Args:
            depth_camera_name: which camera to use for depth. If None, uses the first camera.
            depth_image_size: resize depth images to this size for VAE encoding (square).
        """
        super().__init__(*args, **kwargs)
        self.depth_camera_name = depth_camera_name or self.camera_names[0]
        self.depth_image_size = depth_image_size

        # Transform for depth target images: resize and normalize to [-1, 1]
        self.depth_transform = transforms.Compose([
            transforms.Resize(
                (depth_image_size, depth_image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    def _load_depth_from_h5(self, file_path, step_id):
        """Load a depth image from HDF5 file.

        Raises RuntimeError if depth data is missing.

        Returns:
            depth_pil: PIL Image (3-channel colorized depth)
        """
        with h5py.File(file_path, "r") as f:
            if "depth_images" not in f["observations"]:
                raise RuntimeError(
                    f"HDF5 file {file_path} does not contain 'observations/depth_images'. "
                    f"Joint training requires depth data for every sample."
                )
            depth_group = f["observations"]["depth_images"]
            if self.depth_camera_name not in depth_group:
                raise RuntimeError(
                    f"HDF5 file {file_path} does not contain depth for camera "
                    f"'{self.depth_camera_name}'. Available: {list(depth_group.keys())}"
                )

            depth_raw = depth_group[self.depth_camera_name][step_id]

            # Case 1: JPEG-encoded bytes
            if depth_raw.ndim < 2:
                depth_img = cv2.imdecode(
                    np.frombuffer(depth_raw, np.uint8), cv2.IMREAD_UNCHANGED
                )
            else:
                depth_img = depth_raw

            # Convert to float [0, 1]
            if depth_img.dtype == np.uint16:
                depth_float = depth_img.astype(np.float32) / 65535.0
            elif depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
                depth_float = depth_img.astype(np.float32)
                dmin, dmax = depth_float.min(), depth_float.max()
                if dmax > dmin:
                    depth_float = (depth_float - dmin) / (dmax - dmin)
                else:
                    depth_float = np.zeros_like(depth_float)
            else:
                # uint8 or similar
                depth_float = depth_img.astype(np.float32) / 255.0

            # If multi-channel, take first channel
            if depth_float.ndim == 3:
                depth_float = depth_float[:, :, 0]

            # Invert: near=bright, far=dark (convention for depth prediction)
            depth_float = 1.0 - depth_float

            # Colorize
            colored = colorize_depth(depth_float)
            return Image.fromarray(colored)

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        # Load depth image (required — raises if missing)
        h5_file_path = data_dict["h5_file_path"]
        step_id = data_dict["step_id"]

        depth_pil = self._load_depth_from_h5(h5_file_path, step_id)
        depth_tensor = self.depth_transform(depth_pil)  # (3, H, W), [-1, 1]

        data_dict["depth_image"] = depth_tensor
        return data_dict

    def _collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super()._collate_fn(instances)

        # Collate depth images — every sample must have depth
        depth_images = [inst["depth_image"] for inst in instances]
        batch["depth_images"] = torch.stack(depth_images, dim=0)

        return batch
