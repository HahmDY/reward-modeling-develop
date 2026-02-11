cd $RMOOD_HOME/penaltyrm/rm/dataset/golden

export VLLM_TORCH_COMPILE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python sft_sampling.py --model_name Hahmdong/PRM-llama3.2-3b-alpacafarm-sft \
    --source_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_prompts.json \
    --target_path $RMOOD_HOME/datasets/alpacafarm/rm/rm_sft.json \
    --save_every 1 \
    --gpus 0,1,2,3,4,5,6,7