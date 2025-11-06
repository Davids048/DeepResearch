#!/usr/bin/env python3
"""
Extract trajectories from evolved_kflow.jsonl files.

Given a file with JSONL format where each line contains a JSON object with a 'history' field,
this script extracts trajectories and creates separate files (iter1.jsonl, iter2.jsonl, etc.)
where each file contains the ith trajectory from all input lines.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def extract_trajectories(input_file: Path, output_dir: Path = None):
    """
    Extract trajectories from the input JSONL file and create separate files.

    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to write output files (defaults to same directory as input)
    """
    if output_dir is None:
        output_dir = input_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to hold trajectories for each iteration
    # Key: iteration index, Value: list of trajectories
    trajectories_by_iter = defaultdict(list)

    # Read the input file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue

            # Extract history field
            if 'history' not in data:
                print(f"Warning: 'history' field not found at line {line_num}")
                continue

            history = data['history']
            if not isinstance(history, list):
                print(f"Warning: 'history' is not a list at line {line_num}")
                continue

            # Extract trajectory from each history item
            for i, item in enumerate(history):
                if 'trajectory' not in item:
                    print(f"Warning: 'trajectory' not found in history[{i}] at line {line_num}")
                    continue

                trajectory = item['trajectory']
                trajectories_by_iter[i].append(trajectory)

    # Write trajectories to separate files
    num_iters = len(trajectories_by_iter)
    print(f"Found {num_iters} iteration(s) with trajectories")

    for iter_idx in sorted(trajectories_by_iter.keys()):
        output_file = output_dir / f"iter{iter_idx + 1}.jsonl"
        trajectories = trajectories_by_iter[iter_idx]

        with open(output_file, 'w', encoding='utf-8') as f:
            for trajectory in trajectories:
                json.dump(trajectory, f, ensure_ascii=False)
                f.write('\n')

        print(f"Written {len(trajectories)} trajectories to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract trajectories from evolved_kflow.jsonl files'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to the input JSONL file (e.g., iter1.evolved_kflow.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for extracted trajectory files (defaults to input file directory)'
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None

    extract_trajectories(input_file, output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
