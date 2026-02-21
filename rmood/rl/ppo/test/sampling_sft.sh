cd $RMOOD_HOME

export CUDA_VISIBLE_DEVICES=0
export RM=rm

hf download Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft

python ${RMOOD_HOME}/rmood/rl/ppo/test/sampling.py \
--model_name Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft \
--source_path ${RMOOD_HOME}/datasets/alpacafarm/test/test_prompts.json \
--target_path ${RMOOD_HOME}/datasets/alpacafarm/rl/ppo/RMOOD-qwen3-4b-alpacafarm-sft.json \
--gpus 0 \
--save_every 1
