#!/bin/bash

export WANDB_PROJECT="PenaltyRM"
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $PENALTY_HOME

MODEL_NAME="Hahmdong/PRM-llama3.2-3b-alpacafarm-sft"
TOKENIZING_MODEL="Hahmdong/PRM-llama3.2-3b-alpacafarm-sft"
DATASET_NAME="alpacafarm"
DATA_FILES="$PENALTY_HOME/datasets/${DATASET_NAME}/rm/rm_implicit.jsonl"
NUM_TRAIN_EPOCHS=1
OUTPUT_MODEL_NAME="PRM-llama3.2-3b-${DATASET_NAME}-rm"
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