cd $PENALTY_HOME/penaltyrm/rm/dataset/proxy

export CUDA_VISIBLE_DEVICES=0

python labeling_reward.py \
    --prompts_path $PENALTY_HOME/datasets/alpacafarm/rm/rm_prompts.json \
    --responses_path $PENALTY_HOME/datasets/alpacafarm/rm/rm_sft.json \
    --target_path $PENALTY_HOME/datasets/alpacafarm/rm/rm_rewards.json \
    --model_name Hahmdong/PRM-qwen2.5-14b-alpacafarm-golden-rm \
    --batch_size 32 \
	--num_responses 2