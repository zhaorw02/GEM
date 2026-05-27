import argparse
import os
from train.train import train

from accelerate.logging import get_logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Main script for training RDT with Qwen3VL.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="Path to the configuration file. Default is `configs/base.yaml`.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Enable DeepSpeed and pass the path to its config file or an already initialized DeepSpeed config dictionary",
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
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=8, help="Batch size (per device) for the sampling dataloader."
    )
    parser.add_argument(
        "--num_sample_batches", type=int, default=2, help="Number of batches to sample from the dataset."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_period",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
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
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_period`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Path or name of a pretrained RDT checkpoint to load from.\n"
            "   - a path to a directory, or\n"
            "   - a path to a model checkpoint (*.pt), or\n"
            "   - None to randomly initialize."
        )
    )
    parser.add_argument(
        "--enable_distill",
        action="store_true",
        default=False,
        help="Whether to distill a pre-trained RDT model using DMD distillation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cond_mask_prob",
        type=float,
        default=0.1,
        help="The probability to randomly mask the conditions (except states) during training.",
    )
    parser.add_argument(
        "--cam_ext_mask_prob",
        type=float,
        default=-1.0,
        help="The probability to randomly mask the external camera image during training.",
    )
    parser.add_argument(
        "--state_noise_snr",
        type=float,
        default=None,
        help="The signal-to-noise ratio (SNR, unit: dB) for adding noise to the states.",
    )
    parser.add_argument(
        "--auto_adjust_image_brightness",
        action="store_true",
        default=False,
        help="Whether or not to automatically adjust the brightness of the dark input images.",
    )
    parser.add_argument(
        "--image_aug",
        action="store_true",
        default=False,
        help="Whether or not to apply image augmentation to the input images.",
    )
    parser.add_argument(
        "--precomp_lang_embed",
        action="store_true",
        default=False,
        help="Whether or not to use precomputed language embeddings.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="[TensorBoard] log directory.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported: "tensorboard", "wandb", "comet_ml", "all".',
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
        help="Whether to use mixed precision. Choose between fp16 and bf16.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--hdf5_dir", type=str, default=None, help="Path to the hdf5 dataset directory.")

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save more memory by using setting grads to None instead of zero.",
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
