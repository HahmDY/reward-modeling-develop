cd $PENALTY_HOME

export CUDA_VISIBLE_DEVICES=1,2,3,4

policy_models=(
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

reward_models=(
	"Hahmdong/PRM-qwen2.5-3b-alpacafarm-rm"
	"Hahmdong/PRM-qwen2.5-14b-alpacafarm-golden-rm"
)

for policy_model in ${policy_models[@]}; do
	policy_base_name=$(basename $policy_model)
	for reward_model in ${reward_models[@]}; do
		reward_base_name=$(basename $reward_model)
		echo "Processing policy: $policy_base_name with reward: $reward_base_name"
		python penaltyrm/rm/dataset/proxy/labeling_reward.py \
			--prompts_path datasets/alpacafarm/test/test_prompts.json \
			--responses_path datasets/alpacafarm/rl/results/$policy_base_name.json \
			--target_path penaltyrm/rl/test/rewards/${reward_base_name}/${policy_base_name}.json \
			--model_name $reward_model \
			--num_responses 1
	done
done