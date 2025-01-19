#!/bin/bash

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-3B
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=29500 \
    llama_models/scripts/example_chat_completion.py \
    $CHECKPOINT_DIR \
    --max_batch_size=1 \
    --max_seq_len=256 