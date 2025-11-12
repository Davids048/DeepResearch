import os
import json
import sys
from openai import OpenAI


def extract_data(traj1:dict, traj2:dict):
    """
    Extract and process trajectory data to evaluate reviews.

    Args:
        traj1: Trajectory that generated the correct answer
        traj2: Trajectory containing reviews but did not generate the right answer

    Returns:
        tuple: (filtered_messages_traj1, filtered_messages_traj2) - Filtered messages from both trajectories
    """
    # Filter out tool messages from traj1
    filtered_messages_traj1 = [msg for msg in traj1['messages'] if msg.get('role') != 'tool']

    # Filter out tool messages from traj2
    filtered_messages_traj2 = [msg for msg in traj2['messages'] if msg.get('role') != 'tool']

    # Return the filtered trajectories
    return filtered_messages_traj1, filtered_messages_traj2

def eval_review(correct_trajectory, error_trajectory):
    user_prompt = f"""
You are an expert meta reviewer.

# Instructions:
You will be given 2 materials:
- A ground truth correct trajectory: this trajectory led to a correct prediction.
    - This trajectory will contain a key reasoning step
    - The key reasoning step is defined as the last time the agent considered the right candidate, before it started to perform verifications on it. 
- An error trajectory: this trajectory failed to give the right prediction. This trajectory has a "review" section that has some proposed adjustments based on a previous attempt.

Your task is to analyze why the error trajectory failed.
- Analyze the difference between the right and wrong trajectory. 
- Identify the key reasoning step (there is 1) in the correct trajectory that led to the right answer. 
- Check the review (provided by the user in the first round in the error trajectory).
- Analyze if the review contained the key missing reasoning step.

# Output format: 
1. Key reasoning step: 
    - The round where it happened. 
    - The cause: Options: 1. Agent's own memory, 2. Search result. 
2. Key pivot: 
    - The assistant round before the key reasoning step. Usually, the agent made a key pivot from previous reasoning, and led to the key reasoning step. 
2. Judgement: Whether the reviews in the error trajectory contained this key reasoning step. 
3. Judgement: Whether the error trajectory correctly utilized the reviews to get the answer. If not, why. 
4. Judgement: Whether the error trajectory's fixation on the wrong answer was partially due to high occurance words in the review.
5. What kind of reviews you would have generated. 

# Data
## Correct Trajectory:
{correct_trajectory}

## Error Trajectory:
{error_trajectory}
"""

    # Define the structured response format for similarity judgment
    review_judgement_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "review_judgement",
            "schema": {
                "type": "object",
                "properties": {
                    "response":{
                        "type":"string",
                        "description": "a report that contains all the required information. "
                    },
                },
                "required": ["response"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # Initialize OpenAI client with environment variables
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base if api_base else None
    )

    # Call GPT-4o with structured output
    try:
        print(f"Sending request to OpenAI API (model: gpt-5)...", flush=True)
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            # response_format=review_judgement_format,
            timeout=300.0  # Increased timeout to 5 minutes
        )
        print(f"Received response from OpenAI API", flush=True)

        # Extract the judgment from the response
        result = response.choices[0].message.content
        return result

    except Exception as e:
        print(f"Error calling GPT-4o: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Force unbuffered output for real-time logging with tee
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("Script started...", flush=True)

    # Configuration: list of test configs
    # Each config has: correct_file, review_file, and qids to process
    configs = [
        {
            "name": "ref8-1 vs base 1",
            "correct_file": "output/GLM-4.6/browsecomp/20251111-010304/iter2_scored.jsonl",
            "review_file": "output/GLM-4.6/browsecomp/20251111-065751/iter3_scored.jsonl",
            "qids": [15, 29, 43]
        },
        # Add more configs here as needed
        {
            "name": "ref8-1 vs base 2",
            "correct_file": "output/GLM-4.6/browsecomp/20251111-035622/iter1_scored.jsonl",
            "review_file": "output/GLM-4.6/browsecomp/20251111-065751/iter3_scored.jsonl",
            "qids": [11,17]
        },
    ]

    # Process each configuration
    for config in configs:
        print(f"\n{'#'*80}", flush=True)
        print(f"# {config['name']}", flush=True)
        print(f"{'#'*80}", flush=True)
        print(f"Correct file: {config['correct_file']}", flush=True)
        print(f"Review file: {config['review_file']}", flush=True)
        print(f"Question IDs: {config['qids']}", flush=True)
        print(flush=True)

        # Load all trajectories from both files
        print("Loading trajectories...", flush=True)
        with open(config['correct_file'], 'r') as f:
            correct_trajs = [json.loads(line) for line in f]
        print(f"Loaded {len(correct_trajs)} correct trajectories", flush=True)

        with open(config['review_file'], 'r') as f:
            review_trajs = [json.loads(line) for line in f]
        print(f"Loaded {len(review_trajs)} review trajectories", flush=True)

        # Process each question ID
        for qid in config['qids']:
            print(f"\n{'='*80}", flush=True)
            print(f"Processing Question ID: {qid}", flush=True)
            print(f"{'='*80}", flush=True)

            # Extract data for this question (assuming qid corresponds to line index)
            print(f"Extracting data for qid {qid}...", flush=True)
            traj1 = correct_trajs[qid]
            traj2 = review_trajs[qid]

            # Extract the filtered trajectories
            print(f"Filtering trajectories...", flush=True)
            filtered_traj1, filtered_traj2 = extract_data(traj1, traj2)
            print(f"Filtered traj1: {len(filtered_traj1)} messages, traj2: {len(filtered_traj2)} messages", flush=True)

            # Evaluate the review
            print(f"Calling eval_review (this may take a while)...", flush=True)
            result = eval_review(filtered_traj1, filtered_traj2)
            print(f"eval_review completed", flush=True)

            # Print the result
            print(f"\nEvaluation Result:", flush=True)
            print(result, flush=True)
            print(flush=True)