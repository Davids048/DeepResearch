#!/bin/bash

# Test evaluation script for 3-line sample from browsecomp dataset
# This script tests the LLM-based evaluation on a small sample

cd "$(dirname "$0")"

# INPUT=/home/hal-jundas/agent/DeepResearch/inference/output/previous-outputs/tongyi/browsecomp20250925_164807U6isk6Xq
# INPUT=/mnt/sharefs/users/hao.zhang/ds8-agent/OSDI2025/DeepResearch/inference/output/GLM-4.6/browsecomp/20251105-092742
# INPUT=/mnt/sharefs/users/hao.zhang/ds8-agent/OSDI2025/DeepResearch/inference/output/GLM-4.6/browsecomp/20251105-213058
# INPUT=../inference/output/GLM-4.6/browsecomp/20251105-214811
# INPUT=../inference/output/GLM-4.6/browsecomp/20251105-214334
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-011952
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-012308
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-034937
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-040327
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-041538
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-045053
# INPUT=../inference/output/GLM-4.6/browsecomp/20251108-054456
# INPUT=../inference/output/GLM-4.6/browsecomp/20251109-060600
# INPUT=../inference/output/GLM-4.6/browsecomp/20251109-201924
INPUT=../inference/output/GLM-4.6/browsecomp/20251109-235759

python evaluate_deepsearch_official_glm.py \
    --input_folder $INPUT \
    --dataset browsecomp_en_full \
    --restore_result_path $INPUT/summary.jsonl 


echo ""
echo "Test evaluation complete!"
