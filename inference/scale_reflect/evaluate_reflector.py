#!/usr/bin/env python3
"""
Evaluate the effectiveness of the reflector module.
"""

import argparse
import json
import sys
from pathlib import Path
import wandb
import os

# Add parent directory to path to import from evolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolve.reflector import Reflector
from evolve.playbook import Playbook
from evolve.llm import LLMClient
from logger import setup_logging

logger = setup_logging(name=__name__, level=5)


def load_dataset(input_path: str, line_limit: int = None):
    """Load the JSONL dataset."""
    dataset = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if line_limit and i >= line_limit:
                break
            data = json.loads(line.strip())
            dataset.append(data)
    return dataset


def extract_trajectory(item):
    """Extract trajectory from dataset item."""
    trajectory = {
        'question': item.get('question', ''),
        'prediction': item.get('prediction', ''),
        'messages': item.get('messages', []),
        'answer': item.get('answer', ''),
        'termination': item.get('termination', ''),
    }
    return trajectory


def get_ground_truth_correctness(item):
    """
    Get ground truth correctness from the item.
    Returns 'correct' if is_correct==true, 'incorrect' otherwise.
    """
    is_correct = item.get('is_correct', '')
    if is_correct is True or (isinstance(is_correct, str) and is_correct.lower() == 'true'):
        return 'correct'
    else:
        return 'incorrect'

def print_traj_summary(trajectory):
    # Print trajectory info
    messages = trajectory.get('messages', [])
    num_messages = len(messages)

    # Count total words in messages
    total_words = 0
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                total_words += len(content.split())

    termination = trajectory.get('termination', 'N/A')

    print(f"=" * 40)
    print(f"  Trajectory info:")
    print(f"    - Message count: {num_messages}")
    print(f"    - Total words: {total_words}")
    print(f"    - Termination: {termination}")
    print(f"=" * 40)


def main():
    parser = argparse.ArgumentParser(description='Evaluate reflector module')
    parser.add_argument('--input-file', required=True, type=str, help='Path to input JSONL file')
    parser.add_argument('--line-limit', type=int, default=2,
                        help='Maximum number of lines to process (for debugging)')
    parser.add_argument('--model-name', type=str,
                        default='Qwen/Qwen3-235B-A22B-Thinking-2507',
                        help='Model name for the reflector LLM')
    parser.add_argument('--base-url', type=str,
                        default='http://localhost:6000/v1',
                        help='Base URL for the LLM API')

    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_file}")
    print(f"Line limit: {args.line_limit if args.line_limit else 'None (all lines)'}")
    print(f"Model: {args.model_name}")
    print(f"Base URL: {args.base_url}")
    print("-" * 80)

    # Initialize wandb
    wandb_project = os.getenv("WANDB_PROJECT", "scale_reflect")
    wandb_name = os.getenv("WANDB_NAME", f"{Path(args.input_file).name}_limit{args.line_limit}")
    wandb_notes = os.getenv("WANDB_NOTES", "eval reflector baseline")

    run = wandb.init(
        project=wandb_project,
        name=wandb_name,
        notes=wandb_notes,
        config={
            "input_file": args.input_file,
            "line_limit": args.line_limit,
            "model_name": args.model_name,
            "base_url": args.base_url,
        }
    )

    # Print wandb run id for later use
    print(f"WANDB_RUN_ID={run.id}")
    print("-" * 80)

    # Load dataset
    dataset = load_dataset(args.input_file, args.line_limit)
    print(f"Loaded {len(dataset)} examples")
    print("-" * 80)

    # Initialize reflector
    llm = LLMClient(model_name=args.model_name, base_url=args.base_url)
    reflector = Reflector(llm=llm)

    # Initialize empty playbook (as noted in reflector.py, this is TODO)
    playbook = Playbook()

    # Evaluation metrics
    total = 0
    correct_matches = 0  # Reflector says correct, ground truth is correct
    incorrect_matches = 0  # Reflector says incorrect/incomplete, ground truth is incorrect
    false_positives = 0  # Reflector says correct, ground truth is incorrect
    false_negatives = 0  # Reflector says incorrect/incomplete, ground truth is correct

    # Process each item
    for idx, item in enumerate(dataset):
        print(f"\nProcessing example {idx + 1}/{len(dataset)}...")

        # Extract trajectory
        trajectory = extract_trajectory(item)

        # Get ground truth
        ground_truth = get_ground_truth_correctness(item)

        print_traj_summary(trajectory)

        
        try:
            # Get reflection
            reflector_output, reflection_summary = reflector.reflect(
                trajectory=trajectory,
                playbook=playbook
            )

            # Get reflector's judgement
            reflector_judgement = reflector_output.correctness_judgement.lower()

            # Normalize reflector judgement to correct/incorrect
            # "correct" -> correct
            # "incorrect" or "incomplete" -> incorrect
            if reflector_judgement == 'correct':
                reflector_binary = 'correct'
            else:
                reflector_binary = 'incorrect'

            print(f"  Ground truth: {ground_truth}")
            print(f"  Reflector judgement: {reflector_judgement} (binary: {reflector_binary})")

            # Update metrics
            total += 1

            if ground_truth == 'correct' and reflector_binary == 'correct':
                correct_matches += 1
            elif ground_truth == 'incorrect' and reflector_binary == 'incorrect':
                incorrect_matches += 1
            elif ground_truth == 'incorrect' and reflector_binary == 'correct':
                false_positives += 1
            elif ground_truth == 'correct' and reflector_binary == 'incorrect':
                false_negatives += 1

        except Exception as e:
            print(f"  ERROR: Failed to process example {idx + 1}: {e}")
            logger.error(f"Failed to process example {idx + 1}", exc_info=True)
            continue

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total examples processed: {total}")
    print(f"Correct matches (TP): {correct_matches}")
    print(f"Incorrect matches (TN): {incorrect_matches}")
    print(f"False positives (FP): {false_positives}")
    print(f"False negatives (FN): {false_negatives}")

    # Log metrics to wandb
    metrics = {
        "total": total,
        "correct_matches_TP": correct_matches,
        "incorrect_matches_TN": incorrect_matches,
        "false_positives_FP": false_positives,
        "false_negatives_FN": false_negatives,
    }

    if total > 0:
        accuracy = (correct_matches + incorrect_matches) / total
        print(f"\nAccuracy: {accuracy:.2%} ({correct_matches + incorrect_matches}/{total})")
        metrics["accuracy"] = accuracy

        # Precision and Recall
        if (correct_matches + false_positives) > 0:
            precision = correct_matches / (correct_matches + false_positives)
            print(f"Precision: {precision:.2%}")
            metrics["precision"] = precision

        if (correct_matches + false_negatives) > 0:
            recall = correct_matches / (correct_matches + false_negatives)
            print(f"Recall: {recall:.2%}")
            metrics["recall"] = recall

    print("=" * 80)

    # Log final metrics to wandb
    wandb.log(metrics)
    wandb.finish()


if __name__ == '__main__':
    main()
