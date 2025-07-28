#!/bin/bash

# Ultra-light memory training with numerical stability improvements
# Fixed configuration to prevent NaN loss

export CUDA_VISIBLE_DEVICES=1

python src/train_lora.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --data_path data/gsm8k-train-length-perception.json \
    --bf16 True \
    --output_dir ./ckpts/qwen25-3b-gsm8k-response-length-perception-stable \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 25 \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --max_grad_norm 1.0 \
    --seed 42 \
    --remove_unused_columns False