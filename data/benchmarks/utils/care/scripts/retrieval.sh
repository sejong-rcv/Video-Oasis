#!/bin/bash

MODEL_PATH="checkpoints-release/InternVL2-8B-RA"
DATA=didemo

accelerate launch \
    --num_machines=1 \
    --num_processes 8 \
    --machine_rank 0 \
    tasks/retrieval.py \
    --model_path $MODEL_PATH \
    --num_frames 32 \
    --data $DATA