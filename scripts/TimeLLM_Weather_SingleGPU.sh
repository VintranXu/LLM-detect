#!/bin/bash

# 设置本地llama2-7b-hf模型路径
# export LOCAL_LLAMA_PATH="llama-2-7b-hf"  # 替换为你的本地模型路径
export LOCAL_LLAMA_PATH="/home/kemove/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased/"

model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=6

batch_size=2
d_model=768
d_ff=128

comment='TimeLLM-Weather-Local'

# 使用单GPU模式
accelerate launch  --num_processes 4 --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --llm_model BERT \
  --model_comment $comment
