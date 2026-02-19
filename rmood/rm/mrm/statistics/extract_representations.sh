#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Script to extract representations and compute GDA parameters.
#
# Mode 1 — extract from model (default):
#   bash extract_representations.sh <model_path> [data_path] [output_dir] [batch_size] [max_length]
#
# Mode 2 — skip extraction, load pre-existing .npy files:
#   bash extract_representations.sh --load
#   bash extract_representations.sh --load [chosen.npy] [rejected.npy] [message.npy] [output_dir]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RMOOD_HOME="${RMOOD_HOME:-$(cd $SCRIPT_DIR/../../../.. && pwd)}"

REPR_DIR="$RMOOD_HOME/datasets/alpacafarm/rm/representations"

if [ "$1" = "--load" ]; then
    # ── Mode 2: load pre-existing representations ──────────────────────
    MODEL_PATH="${2:-Hahmdong--RMOOD-qwen3-4b-alpacafarm-rm}"
    OUTPUT_DIR="${3:-$REPR_DIR}"

    echo "=================================================="
    echo "Compute GDA Parameters from existing representations"
    echo "=================================================="
    echo "Model Path: $MODEL_PATH"
    echo "Output:   $OUTPUT_DIR"
    echo "=================================================="
    echo ""

    python -m rmood.rm.mrm.statistics.estimate \
        --model_path "$MODEL_PATH" \
        --load_representations \
        --output_dir "$OUTPUT_DIR" \
        --compute_gda
else
    # ── Mode 1: extract representations from model ─────────────────────
    MODEL_PATH="${1:-Hahmdong/RMOOD-qwen3-4b-alpacafarm-rm}"
    DATA_PATH="${2:-$RMOOD_HOME/datasets/alpacafarm/rm/rm_implicit.jsonl}"
    OUTPUT_DIR="${3:-$REPR_DIR}"
    BATCH_SIZE="${4:-16}"
    MAX_LENGTH="${5:-2048}"

    echo "=================================================="
    echo "Extract Representations from reward model"
    echo "=================================================="
    echo "Model Path: $MODEL_PATH"
    echo "Data Path:  $DATA_PATH"
    echo "Output Dir: $OUTPUT_DIR"
    echo "Batch Size: $BATCH_SIZE"
    echo "Max Length: $MAX_LENGTH"
    echo "=================================================="
    echo ""

    python -m rmood.rm.mrm.statistics.estimate \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --compute_gda
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Done!"
    echo "=================================================="
    echo "Output: $OUTPUT_DIR"
    echo "- chosen_representations.npy"
    echo "- rejected_representations.npy"
    echo "- message_representations.npy"
    echo "- gda_parameters.npz"
    echo "=================================================="
else
    echo "Error: failed!"
    exit 1
fi
