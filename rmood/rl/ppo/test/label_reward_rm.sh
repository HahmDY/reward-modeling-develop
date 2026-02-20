cd $RMOOD_HOME

export CUDA_VISIBLE_DEVICES=1,2,3,4

policy_models=(
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-10"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-20"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-30"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-40"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-50"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-60"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-70"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-80"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-90"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm-ppo-step-100"
)

reward_models=(
	"Skywork/Skywork-Reward-V2-Llama-3.1-8B"
	"Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm"
)

for policy_model in ${policy_models[@]}; do
	policy_base_name=$(basename $policy_model)
	for reward_model in ${reward_models[@]}; do
		reward_base_name=$(basename $reward_model)
		echo "Processing policy: $policy_base_name with reward: $reward_base_name"
		python rmood/rm/dataset/labeling_reward.py \
			--prompts_path $RMOOD_HOME/datasets/alpacafarm/test/test_prompts.json \
			--responses_path $RMOOD_HOME/datasets/alpacafarm/rl/ppo/$policy_base_name.json \
			--target_path $RMOOD_HOME/rmood/rl/ppo/test/rewards/${reward_base_name}/${policy_base_name}.json \
			--model_name $reward_model \
			--num_responses 1
	done
done