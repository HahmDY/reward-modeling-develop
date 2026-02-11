#!/bin/bash

export WANDB_PROJECT="PenaltyRM"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd $PENALTY_HOME

MODEL_NAME="Hahmdong/PRM-qwen2.5-14b-alpacafarm-golden-sft"
TOKENIZING_MODEL="Hahmdong/PRM-qwen2.5-14b-alpacafarm-golden-sft"
DATASET_NAME="alpacafarm"
DATA_FILES="$PENALTY_HOME/datasets/${DATASET_NAME}/rm/gold_rm_implicit.jsonl"
NUM_TRAIN_EPOCHS=1
OUTPUT_MODEL_NAME="PRM-qwen2.5-14b-${DATASET_NAME}-golden-rm"
REWARD_MODEL_TYPE="rm"

LEARNING_RATE=5e-5
python $PENALTY_HOME/penaltyrm/rm/train.py \
    --model_name $MODEL_NAME \
    --tokenizing_model $TOKENIZING_MODEL \
    --data_files $DATA_FILES \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
	--output_model_name $OUTPUT_MODEL_NAME \
    --reward_model_type $REWARD_MODEL_TYPE \
    --learning_rate $LEARNING_RATE