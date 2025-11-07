#!/bin/bash

# Get the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go to parent dir (inference)
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$INFERENCE_DIR"

# Add inference directory to PYTHONPATH
export PYTHONPATH="$INFERENCE_DIR:$PYTHONPATH"

# Set variables for the script parameters
INPUT_FILE="output/GLM-4.6/browsecomp/20251105-171414/iter1.evolved_kflow.jsonl"
START_INDEX=0
END_INDEX=11
PROMPT_VERSION="v2"
NUM_REFLECTIONS=16

# Launch the python script with the specified parameters
python3 evolve/test_reflection_diversity.py \
    --input_file "$INPUT_FILE" \
    --start_index "$START_INDEX" \
    --end_index "$END_INDEX" \
    --prompt_version "$PROMPT_VERSION" \
    --num_reflections "$NUM_REFLECTIONS"
