dataset_name=alpacafarm
reward_model_name=rm

for step in 40; do
cd ~
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /root/checkpoints/penaltyrm/qwen2.5-3b-${reward_model_name}-ppo/global_step_${step}/actor \
    --target_dir PRM-qwen2.5-3b-${dataset_name}-${reward_model_name}-ppo-step-${step}
cd ~/PRM-qwen2.5-3b-${dataset_name}-${reward_model_name}-ppo-step-${step}
huggingface-cli upload Hahmdong/PRM-qwen2.5-3b-${dataset_name}-${reward_model_name}-ppo-step-${step} .
done