#!/bin/bash

# ==================== Network settings (adjust for your cluster) ====================
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,...
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO

CUR_TIME=$(date +%Y%m%d_%H%M%S)

ENV_TYPE="demo"
TASK_NAME="pick_cube_letter"
WANDB_PROJECT="rdt2-qwen3vl-${ENV_TYPE}-${TASK_NAME}"
HDF5_DIR="/path/to/your/${ENV_TYPE}/${TASK_NAME}/processed_rdt2"
OUTPUT_DIR="./checkpoints/${WANDB_PROJECT}/$CUR_TIME/"

# Qwen3VL 2B as the vision-language backbone (jointly trained)
VLM_NAME="Qwen/Qwen3-VL-2B-Instruct"

mkdir -p "./logs/${WANDB_PROJECT}"
LOGGING_DIR="./logs/${WANDB_PROJECT}"
LOGGING_FILE="${LOGGING_DIR}/pretrain.log"

# Batch sizes — reduce if OOM due to joint VLM training
TRAIN_BATCH_SIZE=16
SAMPLE_BATCH_SIZE=16

if [ -f "$LOGGING_FILE" ]; then
    rm "$LOGGING_FILE"
    echo "Log file '$LOGGING_FILE' deleted"
else
    echo "Log file '$LOGGING_FILE' does not exist"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For running in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

# For running in a multi-node/machine setup with DeepSpeed
# deepspeed --hostfile=hostfile.txt main.py \
#     --deepspeed="./configs/zero2.json" \

deepspeed main.py \
    --deepspeed="./configs/zero2.json" \
    --config_path="./configs/${ENV_TYPE}.yaml" \
    --pretrained_vision_language_model_name_or_path=$VLM_NAME \
    --output_dir=$OUTPUT_DIR \
    --hdf5_dir=$HDF5_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=1000000 \
    --checkpointing_period=5000 \
    --sample_period=2500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --state_noise_snr=40 \
    --report_to=wandb 2>&1 | tee -a "$LOGGING_FILE"

    # --precomp_lang_embed \  # NOT used — Qwen3VL encodes text+image jointly
    # --auto_adjust_image_brightness \
    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-1000" \
