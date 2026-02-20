cd $RMOOD_HOME

export CUDA_VISIBLE_DEVICES=0
export RM=rm

# SFT

python ${RMOOD_HOME}/rmood/rl/ppo/test/sampling.py \
	--model_name Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft \
	--source_path ${TAMPERING_HOME}/datasets/hhrlhf/rl/hhrlhf_RL_500_test.json \
	--target_path ${TAMPERING_HOME}/datasets/hhrlhf/rl/responses/hhrlhf_RL_500_test_sft_ver17.json \
	--gpus 0 \
	--save_every 1