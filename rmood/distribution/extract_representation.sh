cd $RMOOD_HOME/rmood/distribution

export CUDA_VISIBLE_DEVICES=0

python -m rmood.distribution.extract_representation \
    --model_name_sampling "Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft" \
    --model_name_rm "Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm" \
    --indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
    --batch_size 64