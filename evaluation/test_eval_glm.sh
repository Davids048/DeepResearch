#!/bin/bash

# Test evaluation script for 3-line sample from browsecomp dataset
# This script tests the LLM-based evaluation on a small sample

cd "$(dirname "$0")"

# INPUT=/home/hal-jundas/agent/DeepResearch/inference/output/previous-outputs/tongyi/browsecomp20250925_164807U6isk6Xq
INPUT=/mnt/sharefs/users/hao.zhang/ds8-agent/OSDI2025/DeepResearch/inference/debug_output/GLM-4.6/browsecomp/20251104-235359
python evaluate_deepsearch_official_glm.py \
    --input_folder $INPUT \
    --dataset browsecomp_en_full \
    --restore_result_path $INPUT/summary.jsonl 


echo ""
echo "Test evaluation complete!"
