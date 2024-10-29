#!/bin/bash

# Constants
TOPK_OUT_DIR="data/topk"
PATCH_OUT_DIR="data/patching"
BATCH_SIZE=1

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
mkdir -p "$TOPK_OUT_DIR"
mkdir -p "$PATCH_OUT_DIR"

# convert model name by replacing / with -
FORMATTED_MODEL_NAME=$(echo $MODEL_NAME | sed 's/\//-/g')

# Run the Python script
python scripts/topk.py \
    --model_name_or_path="$MODEL_NAME" \
    --model_type="$MODEL_TYPE" \
    --in_file="data/${PERTURBATION}-data.tsv.gz" \
    --out_path="$TOPK_OUT_DIR" \
    --k="$K" \
    --batch_size="$BATCH_SIZE" \
    --perturbation_type="$PERTURBATION"

# Run the Python script
python scripts/patch.py \
    --model_name_or_path="$MODEL_NAME" \
    --model_type="$MODEL_TYPE" \
    --in_file="data/topk/${FORMATTED_MODEL_NAME}_${MODEL_TYPE}_${PERTURBATION}_topk_${K}.tsv" \
    --out_path="$PATCH_OUT_DIR" \
    --k="$K" \
    --batch_size="$BATCH_SIZE" \
    --perturbation_type="$PERTURBATION"
