#!/bin/bash

# Ultra memory-optimized training script for Qwen 2.5 3B with GSM8K length perception data
# Use GPU 1 and minimal memory settings

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.train_lora \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./data/gsm8k-train-length-perception.json \
    --output_dir ./ckpts/qwen25-3b-gsm8k-response-length-perception \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 50 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --model_max_length 512 \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj" "v_proj"