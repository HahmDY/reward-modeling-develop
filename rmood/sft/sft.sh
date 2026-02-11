#!/bin/bash

cd $RMOOD_HOME

export WANDB_PROJECT="RMOOD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python $RMOOD_HOME/penaltyrm/sft/sft.py \
  --model_name Qwen/Qwen3-4B-Base \
  --tokenizing_model Qwen/Qwen3-4B-Instruct-2507 \
  --output_model_name RMOOD-qwen3-4b-alpacafarm-sft \
  --dataset_name alpacafarm \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5