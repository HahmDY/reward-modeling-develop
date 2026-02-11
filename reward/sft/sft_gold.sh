#!/bin/bash

cd $PENALTY_HOME

export WANDB_PROJECT="PenaltyRM"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python $PENALTY_HOME/penaltyrm/sft/sft.py \
  --model_name Qwen/Qwen2.5-14B \
  --tokenizing_model meta-llama/Llama-3.2-3B-Instruct \
  --output_model_name PRM-qwen2.5-14b-alpacafarm-golden-sft \
  --dataset_name alpacafarm \
  --num_train_epochs 1