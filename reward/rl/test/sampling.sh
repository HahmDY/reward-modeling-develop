cd $PENALTY_HOME/penaltyrm/rl/test

export VLLM_TORCH_COMPILE=0

# 여러 모델 정의
models=(
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-10"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-20"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-30"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-40"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-50"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-60"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-70"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-80"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-90"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-100"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-110"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-120"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-130"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-140"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-150"
  "Hahmdong/PRM-qwen2.5-3b-alpacafarm-rl-rm-156"
)

for model in "${models[@]}"; do
    short_name=$(basename "$model")

    target="$PENALTY_HOME/datasets/alpacafarm/rl/results/${short_name}.json"

    python sampling.py \
        --model_name "$model" \
        --source_path "$PENALTY_HOME/datasets/alpacafarm/test/test_prompts.json" \
        --target_path "$target" \
        --save_every 1 \
        --gpus 0,1,2,7
done
