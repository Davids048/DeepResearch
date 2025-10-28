#!/bin/bash

# Configuration
INPUT_FILE="output/previous-outputs/tongyi/browsecomp20250925_164807U6isk6Xq/iter1_head50_scored.jsonl"
LINE_LIMIT=100

# Get the directory of the input file and set log path
INPUT_DIR=$(dirname "$INPUT_FILE")
LOG_FILE="${INPUT_DIR}/debug.log"

# Setup wandb
export WANDB_PROJECT="scale_reflect"
export WANDB_NAME="$(basename $(dirname "$INPUT_FILE")).${INPUT_FILE##*/}_limit${LINE_LIMIT}"
export WANDB_NOTES="eval reflector baseline"


# Run evaluation with log capture
python -u scale_reflect/evaluate_reflector.py \
    --input-file "$INPUT_FILE" \
    --line-limit "$LINE_LIMIT" \
    2>&1 | tee "$LOG_FILE"

# Attach debug.log to wandb run
WANDB_RUN_ID=$(grep -oP '^WANDB_RUN_ID=\K.*' "$LOG_FILE" | tail -n1)

if [ -n "$WANDB_RUN_ID" ] && [ "$WANDB_MODE" != "disabled" ]; then
    echo "Attaching debug.log to existing wandb run ($WANDB_RUN_ID)..."
    python - <<EOF
import wandb, os
run = wandb.init(project=os.getenv("WANDB_PROJECT"),
                 id="${WANDB_RUN_ID}",
                 resume="allow")
wandb.save("${LOG_FILE}")
run.finish()
EOF
else
    echo "Skipping wandb upload (either disabled or no run id found)."
fi
