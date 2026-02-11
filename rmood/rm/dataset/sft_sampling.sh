cd $RMOOD_HOME/rmood/rm/dataset

export VLLM_TORCH_COMPILE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

python sft_sampling.py --model_name Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft \
    --source_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_prompts.json \
    --target_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_sft.json \
    --save_every 1 \
    --gpus 0,1,2,3