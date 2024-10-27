#!/bin/bash

# Constants
OUT_DIR="data/topk"
BATCH_SIZE=256

# Check if required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_name> <model_type> <k> <perturbation>"
    echo "Example: $0 bert-base-uncased bi 1000 TFC1"
    exit 1
fi

MODEL_NAME=$1
MODEL_TYPE=$2
K=$3
PERTURBATION=$4

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"


# Run the Python script
python scripts/topk.py \
    --model_name_or_path="$MODEL_NAME" \
    --model_type="$MODEL_TYPE" \
    --out_path="$OUT_DIR" \
    --k="$K" \
    --batch_size="$BATCH_SIZE" \
    --perturbation="$PERTURBATION"
