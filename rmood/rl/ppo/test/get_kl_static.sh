#!/bin/bash

cd $RMOOD_HOME/rmood/rl/ppo/test

export CUDA_VISIBLE_DEVICES=0

PROMPTS_PATH="$RMOOD_HOME/datasets/alpacafarm/test/test_prompts.json"
RESPONSES_DIR="$RMOOD_HOME/datasets/alpacafarm/rl/ppo"
REF_MODEL="Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft"

# (hf_model_name, response_file_stem)
declare -A MODEL_RESPONSE_MAP=(
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-10"]="RMOOD-qwen3-4b-alpacafarm-rm-step-10"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-20"]="RMOOD-qwen3-4b-alpacafarm-rm-step-20"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-30"]="RMOOD-qwen3-4b-alpacafarm-rm-step-30"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-40"]="RMOOD-qwen3-4b-alpacafarm-rm-step-40"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-50"]="RMOOD-qwen3-4b-alpacafarm-rm-step-50"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-60"]="RMOOD-qwen3-4b-alpacafarm-rm-step-60"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-70"]="RMOOD-qwen3-4b-alpacafarm-rm-step-70"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-80"]="RMOOD-qwen3-4b-alpacafarm-rm-step-80"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-90"]="RMOOD-qwen3-4b-alpacafarm-rm-step-90"
  ["RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-100"]="RMOOD-qwen3-4b-alpacafarm-rm-step-100"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-10"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-10"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-20"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-20"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-30"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-30"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-40"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-40"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-50"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-50"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-60"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-60"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-70"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-70"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-80"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-80"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-90"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-90"
  ["RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-100"]="RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-100"
)

for hf_model in "${!MODEL_RESPONSE_MAP[@]}"; do
  resp_stem="${MODEL_RESPONSE_MAP[$hf_model]}"
  resp_file="$RESPONSES_DIR/${resp_stem}.json"

  if [ ! -f "$resp_file" ]; then
    echo "[SKIP] response file not found: $resp_file"
    continue
  fi

  echo "Model: $hf_model  |  Response: $resp_stem"
  python get_kl_static.py \
    --actor_model "Hahmdong/$hf_model" \
    --ref_model "$REF_MODEL" \
    --prompts_path "$PROMPTS_PATH" \
    --responses_path "$resp_file" \
    --save_dir "kl_static/$hf_model" \
    --loss_agg_mode sequence-mean
done
