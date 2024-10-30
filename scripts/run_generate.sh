#!/bin/bash

# Constants
OUT_DIR="data"

# Check if required arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <perturbation> [--stopwords] [--exact_match]"
    echo "Example: $0 TFC1 --stopwords --exact_match"
    exit 1
fi

# Default values for flags
STOPWORDS=false
EXACT_MATCH=false

# Parse arguments
PERTURBATION=$1
shift
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --stopwords)
            STOPWORDS=true
            ;;
        --exact_match)
            EXACT_MATCH=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Run the Python script with appropriate flags
python scripts/generate_data.py \
    --out_path="$OUT_DIR" \
    --perturbation_type="$PERTURBATION" \
    $( [ "$STOPWORDS" = true ] && echo "--stopwords" ) \
    $( [ "$EXACT_MATCH" = true ] && echo "--exact_match" )
