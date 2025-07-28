#!/bin/bash

# Training script for Qwen 2.5 3B with GSM8K length perception data
# Adapted from original train.sh for response length perception training

python -m src.train_lora \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --data_path ./data/gsm8k-train-length-perception.json \
    --output_dir ./ckpts/qwen25-3b-gsm8k-response-length-perception \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --eval_dataset_size 100 \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_bias "none"