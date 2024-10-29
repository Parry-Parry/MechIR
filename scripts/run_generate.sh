#!/bin/bash

# Constants
OUT_DIR="data"

# Check if required arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <perturbation>"
    echo "Example: $0 TFC1"
    exit 1
fi

PERTURBATION=$1

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Run the Python script
python scripts/generate_data.py \
    --out_path="$OUT_DIR" \
    --perturbation_type="$PERTURBATION"
