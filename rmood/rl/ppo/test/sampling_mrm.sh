cd $RMOOD_HOME

export CUDA_VISIBLE_DEVICES=0
export RM=rm

for step in 10 20 30 40 50 60 70 80 90 100; do
	hf download Hahmdong/RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-${step}

    python ${RMOOD_HOME}/rmood/rl/ppo/test/sampling.py \
	--model_name Hahmdong/RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-${step} \
	--source_path ${RMOOD_HOME}/datasets/alpacafarm/test/test_prompts.json \
	--target_path ${RMOOD_HOME}/datasets/alpacafarm/rl/ppo/RMOOD-qwen3-4b-alpacafarm-mrm-sft-based-ppo-step-${step}.json \
	--gpus 0 \
	--save_every 1
done