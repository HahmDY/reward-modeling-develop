#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Script to extract representations from MRM model
# Usage: bash extract_representations.sh <model_path>

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RMOOD_HOME="${RMOOD_HOME:-$(cd $SCRIPT_DIR/../../../.. && pwd)}"

# Default values
MODEL_PATH="${1:-Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm}"
DATA_PATH="${2:-$RMOOD_HOME/datasets/alpacafarm/rm/rm_implicit.jsonl}"
OUTPUT_DIR="${3:-$RMOOD_HOME/datasets/alpacafarm/rm/representations}"
BATCH_SIZE="${4:-16}"
MAX_LENGTH="${5:-2048}"

echo "=================================================="
echo "Extract Representations from MRM Model"
echo "=================================================="
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "=================================================="
echo ""

# Run the extraction script
python -m rmood.rm.mrm.statistics.estimate \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --compute_gda

echo ""
echo "=================================================="
echo "Extraction completed!"
echo "=================================================="
echo "Representations saved to: $OUTPUT_DIR"
echo "- chosen_representations.npy"
echo "- rejected_representations.npy"
echo "- gda_parameters.npz (if --compute_gda was used)"
echo "=================================================="
