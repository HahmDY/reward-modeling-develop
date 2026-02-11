#!/bin/bash

cd $PENALTY_HOME/penaltyrm/rl/test

export CUDA_VISIBLE_DEVICES=0

models=(
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-10"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-20"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-30"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-40"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-50"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-60"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-70"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-80"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-90"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-100"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-110"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-120"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-130"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-140"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-150"
  "PRM-qwen2.5-3b-alpacafarm-rl-rm-156"
)

for model in "${models[@]}"; do
  model_name=$model
  echo "Model: $model_name"
  python get_kl.py \
    --actor_model Hahmdong/$model \
    --ref_model Hahmdong/PRM-qwen2.5-3b-alpacafarm-sft \
    --conversations_path "$PENALTY_HOME/datasets/alpacafarm/test/test_prompts.json" \
    --save_dir "kl/$model_name" \
    --loss_agg_mode sequence-mean
done