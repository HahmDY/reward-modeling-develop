#!/usr/bin/env bash
set -euo pipefail

: "${RMOOD_HOME:?RMOOD_HOME is not set}"

cd "$RMOOD_HOME/rmood/rl/bon"

run_one () {
  local dataset_name="$1"
  echo "[INFO] Sampling for dataset: $dataset_name"

  python sampling.py \
    --model_name "Hahmdong/RMOOD-qwen3-4b-${dataset_name}-sft" \
    --source_path "$RMOOD_HOME/datasets/${dataset_name}/test/test_prompts.json" \
    --target_path "$RMOOD_HOME/datasets/${dataset_name}/rl/bon/test_sft_sampled.json" \
    --save_every 1 \
    --gpus 0,1,2,3
}

run_one "alpacafarm"