#!/bin/bash

# Script to convert Qwen3ForSequenceClassification to MRM
# Usage: bash convert_to_mrm.sh <base_model_path> [output_path] [hub_repo]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RMOOD_HOME="${RMOOD_HOME:-$(cd $SCRIPT_DIR/../../.. && pwd)}"

# Default values
BASE_MODEL_PATH="${1:-Hahmdong/RMOOD-qwen3-4b-alpacafarm-sft}"
OUTPUT_PATH="${2:-$RMOOD_HOME/models/RMOOD-qwen3-4b-alpacafarm-mrm-sft-based}"
PUSH_TO_HUB="${3:-Hahmdong/RMOOD-qwen3-4b-alpacafarm-mrm-sft-based}"

echo "=================================================="
echo "Convert Qwen3ForSequenceClassification to MRM"
echo "=================================================="
echo "Base Model: $BASE_MODEL_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Push to Hub: $PUSH_TO_HUB"
echo "=================================================="
echo ""

# Run conversion
if [ -n "$PUSH_TO_HUB" ]; then
    python -m rmood.rm.mrm.convert_to_mrm \
        --base_model_path "$BASE_MODEL_PATH" \
        --output_path "$OUTPUT_PATH" \
        --push_to_hub "$PUSH_TO_HUB"
else
    python -m rmood.rm.mrm.convert_to_mrm \
        --base_model_path "$BASE_MODEL_PATH" \
        --output_path "$OUTPUT_PATH"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Conversion completed successfully!"
    echo "=================================================="
    echo "MRM model saved to: $OUTPUT_PATH"
    echo ""
    echo "To load the model:"
    echo "  from transformers import AutoModelForSequenceClassification"
    echo "  model = AutoModelForSequenceClassification.from_pretrained("
    echo "      '$OUTPUT_PATH',"
    echo "      trust_remote_code=True"
    echo "  )"
    echo "=================================================="
else
    echo ""
    echo "Error: Conversion failed!"
    exit 1
fi
