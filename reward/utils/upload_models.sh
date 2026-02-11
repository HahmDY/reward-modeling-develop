#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

pairs=(
    "/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_10/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-10"
    "/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_20/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-20"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_30/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-30"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_40/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-40"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_50/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-50"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_60/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-60"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_70/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-70"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_80/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-80"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_90/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-90"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_100/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-100"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_110/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-110"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_120/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-120"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_130/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-130"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_140/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-140"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_150/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-150"
	"/home/dongyoon/checkpoints/PenaltyRM/PRM-qwen2.5-3b-alpacafarm-rm-ppo/global_step_156/actor PRM-qwen2.5-3b-alpacafarm-rl-rm-156"
)

for pair in "${pairs[@]}"; do
    read -r checkpoint_path target_dir <<< "$pair"

    echo "=== Processing $checkpoint_path → $target_dir ==="

    cd ~
    python -m verl.model_merger merge \
        --backend megatron \
        --local_dir "$checkpoint_path" \
        --target_dir "/home/dongyoon/penaltyrm/models/$target_dir"

    cd ~/penaltyrm/models/"$target_dir"
    huggingface-cli upload "Hahmdong/$target_dir" .

    echo "=== Done: $target_dir ==="
    echo
done
