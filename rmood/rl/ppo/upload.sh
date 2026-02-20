dataset_name=alpacafarm
reward_model_name=mrm-sft-based

for step in 10 20 30 40 50 60 70 80 90 100; do
cd ~
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /root/checkpoints/RMOOD_PPO/RMOOD-qwen3-4b-${dataset_name}-${reward_model_name}-ppo/global_step_${step}/actor \
    --target_dir RMOOD-qwen3-4b-${dataset_name}-${reward_model_name}-ppo-step-${step}
cd ~/RMOOD-qwen3-4b-${dataset_name}-${reward_model_name}-ppo-step-${step}
huggingface-cli upload Hahmdong/RMOOD-qwen3-4b-${dataset_name}-${reward_model_name}-ppo-step-${step} .
done