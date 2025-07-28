#!/bin/bash

# Memory-optimized training script for Qwen 2.5 3B with GSM8K length perception data

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.train_lora \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./data/gsm8k-train-length-perception.json \
    --output_dir ./ckpts/qwen25-3b-gsm8k-response-length-perception \
    --bf16 True \
    --tf32 True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --logging_steps 20 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --model_max_length 1024 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj"