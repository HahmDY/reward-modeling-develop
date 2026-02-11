cd $RMOOD_HOME/rmood/rm/dataset/proxy

export CUDA_VISIBLE_DEVICES=0

python labeling_reward.py \
    --prompts_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_prompts.json \
    --responses_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_sft.json \
    --target_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_rewards.json \
    --model_name Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --batch_size 128 \
	--num_responses 2