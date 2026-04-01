#!/bin/bash
# Transformer Training Script
# ============================

set -e

# Default values
CONFIG="configs/base_config.yaml"
DATA_DIR="data/wmt14"
SAVE_DIR="checkpoints"
NUM_GPUS=1
RESUME=""
FP16=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Transformer Training"
echo "========================================"
echo "Config: $CONFIG"
echo "Data: $DATA_DIR"
echo "Save: $SAVE_DIR"
echo "GPUs: $NUM_GPUS"
echo "========================================"

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p logs

# Set environment variables for distributed training
export OMP_NUM_THREADS=4

if [ $NUM_GPUS -gt 1 ]; then
    echo "Starting distributed training on $NUM_GPUS GPUs..."
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        src/train.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE_DIR" \
        $RESUME \
        $FP16
else
    echo "Starting single GPU training..."
    python src/train.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --save-dir "$SAVE_DIR" \
        $RESUME \
        $FP16
fi

echo "Training completed!"
