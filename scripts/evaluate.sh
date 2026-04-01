#!/bin/bash
# Transformer Evaluation Script
# =============================

set -e

CHECKPOINT="${1:-checkpoints/model_best.pt}"
CONFIG="${2:-configs/base_config.yaml}"
DATA_DIR="${3:-data/wmt14}"

echo "========================================"
echo "Transformer Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Data: $DATA_DIR"
echo "========================================"

python src/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --output translations.txt

echo "Evaluation completed!"
