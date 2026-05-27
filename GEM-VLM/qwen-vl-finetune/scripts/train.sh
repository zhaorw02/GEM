#!/bin/bash
echo "current Python: $(which python3)"
echo "current Python version: $(python3 --version)"

# Distributed training configuration
MASTER_ADDR=$CHIEF_IP 
nnode=1
nrank=$((INDEX%nnode))

echo "nranks: $nrank"
# NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=zzzrw/GEM-2B

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (all training datasets)


datasets=gem
run_name="gem_train" 
output_dir="./checkpoints/$run_name"

 
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --warmup_ratio 0.03 \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --tune_vae False \
    --tune_connector False \
    --tune_generator False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 2359296 \
    --min_pixels 25088 \
    --video_max_frames 32 \
    --video_min_frames 8 \
    --video_max_pixels 4915200 \
    --video_min_pixels 100352 \
    --video_fps 2.0 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
torchrun  --nproc_per_node 8 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=23334 ${entry_file} ${args} 