from datetime import datetime
import os
import argparse
import json
from typing import List, Dict, Any
from evolve.llm import LLMClient
from reflector import Reflector, ReflectorOutput

def parse_args():
    parser = argparse.ArgumentParser(description="Run Reflector Inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file with agent trajectories.")
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name to use for reflection.")
    parser.add_argument("--base_url", type=str, default="http://localhost:6000/v1", help="Base URL for the LLM API.")
    return parser.parse_args()


def main(args):
    llm_client = LLMClient(model_name=args.model_name, base_url=args.base_url)
    reflector = Reflector(llm=llm_client)

    # Setup output file 
    timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
    output_file = os.path.join(
        args.input_file.replace(".jsonl", f".reflection.{timestamp}.jsonl")
    )

    # Process each entry in the input file
    with open(args.input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            entry = json.loads(line)
            question = entry["question"]
            prediction = entry["prediction"]
            messages = entry["messages"]
            ground_truth = entry["answer"]

            reflection = reflector.reflect_single(
                question=question,
                prediction=prediction,
                messages=messages,
            )

            output_entry = {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "reflection": reflection.raw,
            }
            outfile.write(json.dumps(output_entry) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)