#!/bin/bash

# Simplified training script for Qwen 2.5 3B with GSM8K length perception data

python -m src.train_lora \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./data/gsm8k-train-length-perception.json \
    --output_dir ./ckpts/qwen25-3b-gsm8k-response-length-perception \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj"