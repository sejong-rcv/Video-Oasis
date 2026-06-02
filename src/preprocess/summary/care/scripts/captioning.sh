#!/bin/bash

MODEL_PATH="path/to/model"
SAVE_DIR="path/to/save/eval/results"
DATA=carebench

accelerate launch \
    --num_machines=1 \
    --num_processes 8 \
    --machine_rank 0 \
    tasks/captioning.py \
    --config_path data.config \
    --dataset_name $DATA \
    --model_path $MODEL_PATH \
    --save_dir $SAVE_DIR \
    --num_frames 32 \
    --api_endpoint "https://api.deepseek.com/v1" \
    --api_key "your-api-key" \
    --api_model "deepseek-chat" \
    --api_num_worker 64 \
    --evaluate
