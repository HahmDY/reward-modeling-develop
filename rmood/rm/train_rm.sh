#!/bin/bash

export WANDB_PROJECT="RMOOD"
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $RMOOD_HOME

DATASET_NAME="alpacafarm"
MODEL_NAME="Hahmdong/RMOOD-qwen3-4b-${DATASET_NAME}-sft"
TOKENIZING_MODEL="Hahmdong/RMOOD-qwen3-4b-${DATASET_NAME}-sft"
OUTPUT_MODEL_NAME="RMOOD-qwen3-4b-${DATASET_NAME}-rm"
DATA_FILES="$RMOOD_HOME/datasets/${DATASET_NAME}/rm/rm_implicit.jsonl"
NUM_TRAIN_EPOCHS=1
REWARD_MODEL_TYPE="rm"
LEARNING_RATE=1e-5

python $RMOOD_HOME/rmood/rm/train.py \
    --model_name $MODEL_NAME \
    --tokenizing_model $TOKENIZING_MODEL \
    --data_files $DATA_FILES \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
	--output_model_name $OUTPUT_MODEL_NAME \
    --reward_model_type $REWARD_MODEL_TYPE \
    --learning_rate $LEARNING_RATE