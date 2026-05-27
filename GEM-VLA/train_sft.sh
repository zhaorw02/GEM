#!/bin/bash

# ==================== Network settings (adjust for your cluster) ====================
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,...
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO

CUR_TIME=$(date +%Y%m%d_%H%M%S)

ENV_TYPE="demo"
TASK_NAME="unzip_0301"
WANDB_PROJECT="rdt2-qwen3vl-sft-${ENV_TYPE}-${TASK_NAME}"
HDF5_DIR="/path/to/your/${ENV_TYPE}/${TASK_NAME}/processed_rdt2"

# Resume settings: set RESUME=true and optionally RESUME_CHECKPOINT to resume training
RESUME=false
RESUME_CHECKPOINT="latest"  # "latest" or specific e.g. "checkpoint-5000"

CKPT_BASE="./checkpoints/${WANDB_PROJECT}"
if [ "$RESUME" = true ]; then
    LATEST_DIR=$(ls -td "${CKPT_BASE}"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        echo "Error: No previous checkpoint directory found under ${CKPT_BASE}/"
        exit 1
    fi
    OUTPUT_DIR="$LATEST_DIR"
    echo "Resuming from existing output dir: $OUTPUT_DIR"
    RESUME_ARGS="--resume_from_checkpoint=${RESUME_CHECKPOINT}"
else
    OUTPUT_DIR="${CKPT_BASE}/$CUR_TIME/"
    RESUME_ARGS=""
fi

# Qwen3VL 2B SFT checkpoint (without depth)
VLM_NAME="/path/to/your/qwen3vl_sft_checkpoint"

mkdir -p "./logs/${WANDB_PROJECT}"
LOGGING_DIR="./logs/${WANDB_PROJECT}"
LOGGING_FILE="${LOGGING_DIR}/pretrain.log"

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
    --report_to=wandb \
    $RESUME_ARGS \
    2>&1 | tee -a "$LOGGING_FILE"
