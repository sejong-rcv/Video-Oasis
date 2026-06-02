#!/bin/bash

OUTPUT_DIR="checkpoints/e5v-qwen2vl-7b-mix-recap-2ksteps-nli-lr-2e-5-mbs-32-bs-768-llm"
RUN_NAME=`basename $OUTPUT_DIR`

args=()

BASE_MODEL="checkpoints/Qwen2-VL-7B-Mix-Recap-2ksteps"
BATCH_SIZE=768
MICRO_BATCH_SIZE=32
EPOCH=2
LR=2e-5
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=8
NUM_NODES=1

echo $BASE_MODEL
echo $MICRO_BATCH_SIZE $BATCH_SIZE
wandb online

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES tasks/finetuning.py \
        --model_name_or_path $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --warmup_ratio $WARMUP_RATIO \
        --learning_rate $LR \
        --cutoff_len $CUTOFF_LEN \
        --output_dir $OUTPUT_DIR  \
        --run_name $RUN_NAME \
        --use_neg_sentence --save_steps 1000 \
        --deepspeed ds.config \
        --bf16 \
        --logging_steps 1 --grad_checkpoint