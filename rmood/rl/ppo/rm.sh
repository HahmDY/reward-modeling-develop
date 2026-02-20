set -x

train_files=$RMOOD_HOME/datasets/alpacafarm/rl/rl_prompts.parquet
test_files=$RMOOD_HOME/datasets/alpacafarm/val/val_prompts.parquet

policy_model_path=Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft
reward_model_path=Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm

experiment_name=RMOOD-qwen3-4b-alpacafarm-rm-ppo

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1536 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$policy_model_path \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    critic.optim.lr=2e-6 \
    critic.model.path=$policy_model_path \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.megatron.tensor_model_parallel_size=4 \
    critic.megatron.param_offload=True \
    critic.megatron.grad_offload=True \
    critic.megatron.optimizer_offload=True \
    reward_model.enable=True \
    reward_model.model.path=$reward_model_path \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.megatron.tensor_model_parallel_size=4 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='RMOOD_PPO' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@