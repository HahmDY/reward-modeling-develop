#!/bin/bash

cd $RMOOD_HOME

export WANDB_PROJECT="RMOOD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python $RMOOD_HOME/rmood/sft/sft.py \
  --model_name Qwen/Qwen3-4B-Base \
  --tokenizing_model Qwen/Qwen2.5-7B \
  --output_model_name RMOOD-qwen3-4b-alpacafarm-sft \
  --dataset_name alpacafarm \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5