import os
import json
import sys
from openai import OpenAI

def extract_data(traj:dict):
    """
    Extract and process trajectory data to evaluate reviews.

    Args:
        traj: Trajectory containing reviews but did not generate the right answer

    Returns:
        list: Filtered messages from the trajectory
    """
    # Filter out tool messages from traj
    filtered_messages = [msg for msg in traj['messages'] if msg.get('role') != 'tool']

    # Return the filtered trajectory
    return filtered_messages

def eval_review(trajectory):

    user_prompt = f"""
You are an expert meta reviewer.

# Instructions:
You will be given a trajectory that has a "review" section with some proposed adjustments based on a previous attempt.

Your task is to analyze the trajectory:
- Check the review (provided by the user in the first round in the trajectory).
- Analyze how the trajectory utilized or failed to utilize the review.

# Output format:
1. Judgement: Whether the trajectory correctly utilized the reviews to get the answer. If not, why.
2. Judgement: Whether the trajectory's fixation on the wrong answer was partially due to high occurance words in the review.

# Data
## Trajectory:
{trajectory}
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
    # Each config has: review_file and qids to process
    configs = [
        # {
        #     "name": "ref8-1",
        #     "review_file": "output/GLM-4.6/browsecomp/20251111-065751/iter3_scored.jsonl",
        #     "qids": [15, 29, 43]
        # },
        # # Add more configs here as needed
        # {
        #     "name": "ref8-1",
        #     "review_file": "output/GLM-4.6/browsecomp/20251111-065751/iter3_scored.jsonl",
        #     "qids": [11,17]
        # },
        # {
        #     "name": "ref8-1",
        #     "review_file": "output/GLM-4.6/browsecomp/20251111-065751/iter2_scored.jsonl",
        #     "qids": [15, 27,33,39,43,]
        # },
        {
            "name": "ref8",
            "review_file": "output/GLM-4.6/browsecomp/20251112-034329/iter2_scored.jsonl",
        },
    ]

    # Process each configuration
    for config in configs:
        print(f"\n{'#'*80}", flush=True)
        print(f"# {config['name']}", flush=True)
        print(f"{'#'*80}", flush=True)
        print(f"Review file: {config['review_file']}", flush=True)
        print(flush=True)

        # Load all trajectories from review file
        print("Loading trajectories...", flush=True)
        with open(config['review_file'], 'r') as f:
            review_trajs = [json.loads(line) for line in f]
        print(f"Loaded {len(review_trajs)} review trajectories", flush=True)

        # If no qids is provided in the config, use the incorrect ones
        # Each line in the file is a traj, each contains a field: is_correct, it is True when correct, '' when incorrect
        if 'qids' not in config or config['qids'] is None:
            qids = [i for i, traj in enumerate(review_trajs) if not traj.get('is_correct')]
            print(f"No qids provided, using {len(qids)} incorrect trajectories", flush=True)
        else:
            qids = config['qids']
            print(f"Using provided qids: {qids}", flush=True)

        print(f"Question IDs: {qids}", flush=True)
        print(flush=True)

        # Process each question ID
        for qid in qids:
            print(f"\n{'='*80}", flush=True)
            print(f"Processing Question ID: {qid}", flush=True)
            print(f"{'='*80}", flush=True)

            # Extract data for this question (assuming qid corresponds to line index)
            print(f"Extracting data for qid {qid}...", flush=True)
            traj = review_trajs[qid]

            # Extract the filtered trajectory
            print(f"Filtering trajectory...", flush=True)
            filtered_traj = extract_data(traj)
            print(f"Filtered trajectory: {len(filtered_traj)} messages", flush=True)

            # Evaluate the review
            print(f"Calling eval_review (this may take a while)...", flush=True)
            result = eval_review(filtered_traj)
            print(f"eval_review completed", flush=True)

            # Print the result
            print(f"\nEvaluation Result:", flush=True)
            print(result, flush=True)
            print(flush=True)