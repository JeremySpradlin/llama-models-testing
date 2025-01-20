#!/bin/bash

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

# Create timestamp for the output file
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_FILE="chat_session_${TIMESTAMP}.txt"

CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-3B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=29500 \
    chat.py \
    $CHECKPOINT_DIR \
    --max_batch_size=1 \
    --max_seq_len=256 2>&1 | tee "${OUTPUT_FILE}"

echo "Output has been saved to ${OUTPUT_FILE}" 