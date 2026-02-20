#!/bin/bash

cd $RMOOD_HOME/rmood/rl/ppo/test

export CUDA_VISIBLE_DEVICES=0

models=(
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-10"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-20"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-30"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-40"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-50"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-60"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-70"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-80"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-90"
  "RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-100"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-10"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-20"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-30"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-40"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-50"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-60"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-70"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-80"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-90"
  "RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-100"
)

for model in "${models[@]}"; do
  model_name=$model
  echo "Model: $model_name"
  python get_kl.py \
    --actor_model Hahmdong/$model \
    --ref_model Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft \
    --conversations_path "$RMOOD_HOME/datasets/alpacafarm/test/test_prompts.json" \
    --save_dir "kl/$model_name" \
    --loss_agg_mode sequence-mean
done