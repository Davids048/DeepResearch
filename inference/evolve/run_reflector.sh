# get this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/

python run_reflector.py \
    --input_file /home/hal-jundas/agent/DeepResearch/inference/output/Qwen3-235B-A22B-Thinking-2507/browsecomp/20251022-182128/iter1.jsonl \
    --model_name Qwen/Qwen3-235B-A22B-Thinking-2507

