"""Entry point for joint VLM + RDT + Depth training.

Extends main.py with depth-expert-specific arguments.
"""
import argparse
import os

from accelerate.logging import get_logger
from train.train_joint import train


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Joint training: VLM + Action Expert (RDT) + Depth Expert (SANA)."
    )

    # ==================== Shared args (same as main.py) ====================
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config file.",
    )
    parser.add_argument(
        "--pretrained_vision_language_model_name_or_path",
        type=str,
        default=None,
        help="Pretrained vision language model name or path (e.g., Qwen/Qwen3-VL-2B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory for model predictions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the sampling dataloader.",
    )
    parser.add_argument(
        "--num_sample_batches",
        type=int,
        default=2,
        help="Number of batches to sample from the dataset.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. Overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_period",
        type=int,
        default=500,
        help="Save a checkpoint every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume from a checkpoint. Use "latest" for the last available.',
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path or name of a pretrained RDT checkpoint.",
    )
    parser.add_argument(
        "--enable_distill",
        action="store_true",
        default=False,
        help="Whether to distill a pre-trained RDT model using DMD.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--cond_mask_prob",
        type=float,
        default=0.1,
        help="Probability to randomly mask conditions during training.",
    )
    parser.add_argument(
        "--cam_ext_mask_prob",
        type=float,
        default=-1.0,
        help="Probability to randomly mask external camera image.",
    )
    parser.add_argument(
        "--state_noise_snr",
        type=float,
        default=None,
        help="SNR (dB) for adding noise to states.",
    )
    parser.add_argument(
        "--auto_adjust_image_brightness",
        action="store_true",
        default=False,
        help="Auto adjust brightness of dark input images.",
    )
    parser.add_argument(
        "--image_aug",
        action="store_true",
        default=False,
        help="Apply image augmentation to input images.",
    )
    parser.add_argument(
        "--precomp_lang_embed",
        action="store_true",
        default=False,
        help="Use precomputed language embeddings.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale learning rate by GPUs, grad accum steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Scheduler type: "linear", "cosine", "constant", etc.',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", type=float, default=1.0, help="Power factor of polynomial scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam beta1."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam beta2."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Adam epsilon."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to Hub."
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="Token for Hub."
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None, help="Hub repository name."
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs", help="TensorBoard log directory."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Report to: "tensorboard", "wandb", "comet_ml", "all".',
    )
    parser.add_argument(
        "--sample_period",
        type=int,
        default=-1,
        help="Run sampling every X steps.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training."
    )
    parser.add_argument(
        "--hdf5_dir", type=str, default=None, help="Path to HDF5 dataset directory."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Set grads to None instead of zero.",
    )

    # ==================== Depth expert args ====================
    parser.add_argument(
        "--sana_pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained SANA model (with transformer/ and vae/ subdirs).",
    )
    parser.add_argument(
        "--pretrained_depth_expert_path",
        type=str,
        default=None,
        help="Path to a pretrained depth expert checkpoint (sana + connector weights).",
    )
    parser.add_argument(
        "--depth_loss_weight",
        type=float,
        default=0.1,
        help="Weight for depth loss in the combined loss: total = action + weight * depth.",
    )
    parser.add_argument(
        "--depth_drop_prob",
        type=float,
        default=0.1,
        help="Probability to drop depth conditioning for classifier-free guidance.",
    )
    parser.add_argument(
        "--depth_image_size",
        type=int,
        default=256,
        help="Resize depth target images to this square size for VAE encoding.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
