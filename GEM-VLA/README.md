# VLA Training with GEM

This repository contains the training code for the **RDT2 Action Expert** — a flow-matching diffusion transformer conditioned on vision-language model (VLM) hidden states for robot manipulation. It supports:

- **VLM + RDT joint training**: GEM / Qwen3-VL as the vision-language backbone, with an RDT action expert that takes multi-layer hidden states via cross-attention.
- **Depth expert** (optional): The depth prediction module jointly trained with the action expert for improved spatial understanding.
- **Distillation**: DMD-based single-step distillation for faster inference.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [VLM + RDT Training (from scratch)](#vlm--rdt-training-from-scratch)
  - [VLM + RDT SFT (from checkpoint)](#vlm--rdt-sft-from-checkpoint)
  - [Joint Training with Depth Expert](#joint-training-with-depth-expert)
- [Inference](#inference)
- [Project Structure](#project-structure)

## Overview

The architecture consists of three components:

1. **GEM-2B / Qwen3-VL-2B** — Vision-language model that encodes images and language instructions. Multi-layer hidden states are extracted for the action expert.
2. **RDT (Robotics Diffusion Transformer)** — A flow-matching diffusion transformer that predicts action chunks conditioned on VLM hidden states and proprioceptive state.
3. **Depth Expert** — Optional depth prediction module conditioned on VLM vision tokens, providing auxiliary depth supervision during training.

## Installation

```bash
conda create -n rdt2 python=3.10 -y
conda activate rdt2

# Install PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install flash attention
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install transformers accelerate deepspeed diffusers
pip install h5py pyyaml tqdm wandb imgaug opencv-python timm pillow
pip install huggingface_hub
```

## Data Preparation

### 1. Prepare raw episode data

Each episode should be a directory containing:
- `raw_actions.npy` — Raw action data, shape `(N, action_dim)`
- `left_stereo/` — Directory of left camera PNG images (`0.png`, `1.png`, ...)
- `right_stereo/` — Directory of right camera PNG images

Optionally, for depth training, run [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V2) on the camera directories to generate `{frame_id}_da3.png` files.

### 2. Prepare episode-to-instruction JSON

Create a JSON file mapping episode directory names to language instructions:

```json
{
    "episode_001": "Pick up the red cube.",
    "episode_002": "Pick up the red cube.",
    ...
}
```

### 3. Generate HDF5 files

```bash
python data/generate_hdf5.py \
    --task_name pick_cube_letter \
    --root_dir /path/to/raw_episodes \
    --target_dir /path/to/output/processed_rdt2 \
    --epsd_json /path/to/episode_instructions.json \
    --config_path configs/demo.yaml

# Skip depth images:
python data/generate_hdf5.py ... --no-depth

# For Franka robot data (swaps left/right cameras):
python data/generate_hdf5.py ... --is-franka
```

The output HDF5 structure:
```
episode_0.hdf5
├── action                     # (N, action_dim), float32
└── observations/
    ├── images/
    │   ├── left_stereo        # (N, H, W, 3), uint8
    │   └── right_stereo       # (N, H, W, 3), uint8
    ├── depth_images/          # (optional)
    │   ├── left_stereo        # (N, H, W, 3), uint8
    │   └── right_stereo       # (N, H, W, 3), uint8
    └── state                  # (N, state_dim), float32
```

## Training

### VLM + RDT Training (from scratch)

Train Qwen3-VL-2B jointly with an RDT action expert from scratch:

```bash
bash train.sh
```

Key arguments to modify in `train.sh`:
- `HDF5_DIR` — Path to your processed HDF5 dataset
- `VLM_NAME` — Pretrained VLM name (default: `Qwen/Qwen3-VL-2B-Instruct`)
- `TRAIN_BATCH_SIZE` — Batch size per GPU
- `--max_train_steps` — Total training steps
- `--learning_rate` — Learning rate

### VLM + RDT SFT (from checkpoint)

Fine-tune from a pretrained checkpoint:

```bash
bash train_sft.sh
```

Modify `VLM_NAME` to point to your pretrained VLM checkpoint.

### Joint Training with Depth Expert

Train VLM + RDT + depth expert jointly:

```bash
bash train_joint.sh
```

Additional arguments:
- `DEPTH_PATH` — Path to pretrained depth prediction model (extracted from [GEM-2B]())
- `DEPTH_WEIGHT` — Weight for depth loss (default: 0.1)

## Inference

```python
import yaml
import numpy as np
from models.rdt_inferencer import RDTInferencer

config = yaml.safe_load(open("configs/demo.yaml", "r"))

model = RDTInferencer(
    config=config,
    pretrained_path="path/to/rdt_checkpoint",
    normalizer_path="path/to/normalizer.pt",
    pretrained_vision_language_model_name_or_path="path/to/vlm_checkpoint",
    device="cuda:0",
)

observations = {
    "images": {
        "left_stereo": np.zeros((384, 384, 3), dtype=np.uint8),   # left camera
        "right_stereo": np.zeros((384, 384, 3), dtype=np.uint8),  # right camera
    },
    "state": np.zeros(14, dtype=np.float32),
}

action_chunk = model.step(observations, instruction="Pick up the cube.")
# action_chunk: torch.Tensor of shape (action_chunk_size, action_dim)
```

## Project Structure

```
.
├── configs/
│   ├── demo.yaml              # Model & dataset config
│   └── zero2.json             # DeepSpeed ZeRO-2 config
├── data/
│   └── generate_hdf5.py       # Data preprocessing script
├── models/
│   ├── rdt/                   # RDT transformer architecture
│   │   ├── model.py           # Main RDT model
│   │   ├── blocks.py          # Transformer blocks (self-attn, cross-attn, FFN)
│   │   ├── attention.py       # Attention with GQA and flash attention
│   │   ├── norm.py            # RMSNorm
│   │   └── pos_emb.py         # Sinusoidal positional embeddings
│   ├── multimodal_encoder/    # Vision encoders (DINOv2 + SigLIP)
│   ├── normalizer/            # Action normalization
│   ├── rdt_runner.py          # RDT flow-matching training/inference
│   ├── rdt_distill_runner.py  # DMD distillation runner
│   ├── rdt_inferencer.py      # End-to-end inference wrapper
│   ├── depth_expert.py        # depth expert module
│   ├── ema_model.py           # Exponential moving average
│   └── hub_mixin.py           # HuggingFace Hub save/load
├── train/
│   ├── train.py               # VLM + RDT training loop
│   ├── train_joint.py         # VLM + RDT + Depth joint training loop
│   ├── dataset.py             # HDF5 dataset for action training
│   ├── dataset_joint.py       # HDF5 dataset with depth images
│   ├── sample.py              # Evaluation sampling
│   └── image_corrupt.py       # Image augmentation
├── main.py                    # Entry point for VLM + RDT training
├── main_joint.py              # Entry point for joint training with depth
├── train.sh                   # Training from scratch
├── train_sft.sh               # SFT from checkpoint
├── train_sft_w_depth.sh       # SFT with depth from checkpoint
└── train_joint.sh             # Joint training with depth expert
```
