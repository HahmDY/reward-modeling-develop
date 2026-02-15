cd $RMOOD_HOME/rmood/distribution

python -m rmood.distribution.sampling \
  --model_name "Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft" \
  --source_path "${RMOOD_HOME}/datasets/alpacafarm/rm/rm_prompts.json" \
  --indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19" \
  --num_responses 512 \
  --gpus "0"