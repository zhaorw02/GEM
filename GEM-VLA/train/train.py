#!/usr/bin/env python
# coding=utf-8
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from models.rdt_runner import RDTRunner
from models.rdt_distill_runner import RDTDistillRunner
from train.dataset import H5VLADataset
from train.sample import log_sample_res


class JointVLMRDTModel(torch.nn.Module):
    """Wrapper that combines VLM and RDT into a single nn.Module
    so DeepSpeed can manage them together."""
    def __init__(self, vlm, rdt):
        super().__init__()
        self.vlm = vlm
        self.rdt = rdt


if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
---
    """
    model_card = f"""
# RDT - {repo_id}

This is a RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Initialize Qwen3VL 2B as the vision-language encoder
    processor = AutoProcessor.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    vision_language_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        torch_dtype=weight_dtype,
        attn_implementation="flash_attention_2",
    )
    # NOTE: Do NOT call .eval() — VLM trains jointly with RDT

    # Validate selected_layers config
    if isinstance(config["model"]["selected_layers"], list):
        assert len(config["model"]["selected_layers"]) == config["model"]["rdt"]["depth"], (
            f"selected_layers ({config['model']['selected_layers']}) must be a list of integers with length "
            f"{config['model']['rdt']['depth']}"
        )
    elif not isinstance(config["model"]["selected_layers"], int):
        raise ValueError(
            f"selected_layers ({config['model']['selected_layers']}) must be a list of integers or an integer"
        )

    # Create RDT config
    # lang_token_dim is set when lang_adaptor is enabled,
    # to project VLM hidden states to RDT hidden_size.
    rdt_config = {
        "state_dim": config["common"]["state_dim"],
        "action_dim": config["common"]["action_dim"],
        "pred_horizon": config["common"]["action_chunk_size"],
        "config": config["model"],
        "act_pos_emb_config": [
            ('action', config["common"]["action_chunk_size"]),
            # Learnable register tokens
            ('register', config["model"]["rdt"]["num_register_tokens"]),
        ],
        "dtype": weight_dtype,
    }

    if config["model"].get("lang_adaptor", None) is not None:
        rdt_config.update({
            "lang_token_dim": config["model"]["lang_token_dim"],
        })

    # Load from a pretrained checkpoint
    if (
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Loading model from a pretrained checkpoint directory.")
        rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        rdt = RDTRunner(**rdt_config)

        if (
            args.resume_from_checkpoint is None
            and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)
        ):
            # Since EMA is deprecated, we do not load EMA from the pretrained checkpoint
            logger.info("Loading model from a pretrained checkpoint file.")
            checkpoint = torch.load(args.pretrained_model_name_or_path)
            rdt.load_state_dict(checkpoint["module"])
        else:
            logger.info("Constructing model from scratch.")

    # Initialize the distillation model
    if args.enable_distill:
        if args.pretrained_model_name_or_path is None:
            raise ValueError("When distillation is enabled, `pretrained_model_name_or_path` must be provided.")
        logger.info("Constructing distillation model from pretrained checkpoint.")
        rdt_ditill = RDTDistillRunner(
            rdt_config=rdt_config,
            dtype=weight_dtype,
        )
        rdt_ditill.initialize_from_pretrained_denoiser(rdt)
        rdt = rdt_ditill    # in-place replace the original RDT with the distillation model

    # Wrap VLM + RDT into a single module for DeepSpeed compatibility
    joint_model = JointVLMRDTModel(vision_language_model, rdt)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, JointVLMRDTModel):
                    model_to_save.rdt.save_pretrained(output_dir)
                    vlm_save_dir = os.path.join(output_dir, "vision_language_model")
                    model_to_save.vlm.save_pretrained(vlm_save_dir)
                    processor.save_pretrained(vlm_save_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation — joint_model contains both VLM and RDT
    optimizer = optimizer_class(
        joint_model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = H5VLADataset(
        config=config,
        hdf5_dir=args.hdf5_dir,
        tokenizer=processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        auto_adjust_image_brightness=args.auto_adjust_image_brightness,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
    )
    sample_dataset = H5VLADataset(
        config=config,
        hdf5_dir=args.hdf5_dir,
        tokenizer=processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        auto_adjust_image_brightness=args.auto_adjust_image_brightness,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset._collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=sample_dataset._collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # A single JointVLMRDTModel wraps both VLM and RDT for DeepSpeed compatibility.
    joint_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        joint_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("rdt-2", config=vars(args))
        accelerator.init_trackers(os.getenv("WANDB_PROJECT", "rdt-2"), config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
            except:
                # load deepspeed's state_dict
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                joint_model.module.load_state_dict(checkpoint["module"])

            # load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        joint_model.train()

        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        # Forward and backward...
        for batch in train_dataloader:
            with accelerator.accumulate(joint_model):
                actions = batch["actions"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)

                # Access sub-models via the unwrapped joint_model
                unwrapped = accelerator.unwrap_model(joint_model)

                # Qwen3VL forward — WITH gradients for joint training
                lang_attn_mask = batch["vision_language_model_inputs"]["attention_mask"].to(dtype=torch.bool)
                vlm_outputs = unwrapped.vlm(
                    **batch["vision_language_model_inputs"],
                    output_hidden_states=True,
                    use_cache=False,
                )

                # Extract per-layer hidden states for RDT cross-attention
                # hidden_states: tuple of (num_layers+1) tensors, each (B, seq_len, vlm_hidden_size)
                selected_layers = config["model"]["selected_layers"]
                if isinstance(selected_layers, list):
                    lang_hidden = torch.stack(
                        [vlm_outputs.hidden_states[i] for i in selected_layers],
                        dim=1,
                    )  # (B, depth, seq_len, vlm_hidden_size)
                else:
                    lang_hidden = vlm_outputs.hidden_states[selected_layers]
                    # (B, seq_len, vlm_hidden_size)

                loss = unwrapped.rdt(
                    lang_tokens=lang_hidden,
                    lang_attn_mask=lang_attn_mask,
                    state_tokens=states,
                    action_gt=actions,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(joint_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    unwrapped = accelerator.unwrap_model(joint_model)
                    sample_loss_for_log = log_sample_res(
                        unwrapped.vlm,
                        config["model"]["selected_layers"],
                        unwrapped.rdt,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.enable_distill:
                unwrapped = accelerator.unwrap_model(joint_model)
                logs.update(unwrapped.rdt.get_loss_dict())
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(joint_model)
        unwrapped.rdt.save_pretrained(args.output_dir)
        # Save the jointly-trained VLM
        vlm_save_dir = os.path.join(args.output_dir, "vision_language_model")
        unwrapped.vlm.save_pretrained(vlm_save_dir)
        processor.save_pretrained(vlm_save_dir)

        logger.info(f"Saved Model to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
                # ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
