import os
import random
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from evolve.test_diversity_prompts import (
    REFLECTION_SYSTEM_PROMPTS,
    REFLECTION_USER_TEMPLATES,
    REFLECTION_TOOLS
)
from evolve.utils import get_glm_openai_client, get_glm_tokenizer
from parse_tools_utils import parse_model_response
from prompt_builder import tools_plain2openai
from logger import setup_logging

logger = setup_logging(name=__name__, level=10)


def generate_reflections(
    trajectory:dict,
    reflector_system_prompt,
    reflector_user_template,
    reflector_tools_plain,
):
    messages = trajectory["messages"]
    prediction = trajectory["prediction"] 
    reflector_user_prompt = reflector_user_template.format(
        messages = messages,
        prediction = prediction,
    )
    tokenizer = get_glm_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": reflector_system_prompt},
            {"role": "user", "content": reflector_user_prompt},
        ],
        tools = tools_plain2openai(reflector_tools_plain),
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    client = get_glm_openai_client()
    def _generate_single_reflection():
        response = client.completions.create(
            model="zai-org/GLM-4.6",
            prompt = prompt,
            # Use following params for more variety.
            temperature=1.2,
            top_p = 0.85,
            seed = random.randint(1,1000),
            max_tokens=16000,
            # **DEFAULT_COMPLETION_CONFIG,
        )
        response = response.choices[0].text
        response = "<think>" + response
        logger.debug(f">>>>>>>>>> reflector response:{response}.")

        parsed_response = None
        try:
           # GLM is using pure text handling.
           parsed_response = parse_model_response(response, reflector_tools_plain)
           reasoning_content = parsed_response.get("reasoning_content", "")
           tool_calls = parsed_response.get("tool_calls", [])
           data = tool_calls[0]["arguments"]
           data["reasoning_content"] = reasoning_content

           logger.debug(f"reasoning_content: {reasoning_content}, toolcalls: {tool_calls}")
        except Exception as e:
            logger.error(f"Unexpected error parsing reflector response, using fallback output. Error: {e}. Raw response:{parsed_response}")
            # Create a fallback data object for unexpected errors
            data = {
                "error": "Reflector encountered unexpected error",
            }
        return data 
    reflections = []
    num_reflections = int(os.getenv("MAX_REFLECTIONS", 16))           
    num_reflection_workers = int(os.getenv("MAX_REFLECTION_WORKERS", 16))           
    with ThreadPoolExecutor(max_workers = num_reflection_workers) as executor:
        futures = [executor.submit(_generate_single_reflection) for _ in range(num_reflections)]
        for future in as_completed(futures):
            try:
                reflections.append(future.result())
            except Exception as e:
                logger.error(f"Reflection future failed: {e}.")
                reflections.append({"error": "Reflection failed"})
    # Aggregate the results - pickout the ones where reflector judge the trace as wrong. 
    error_reflections = []
    for reflection in reflections:
        verdict = reflection.get("correctness_judgement", "")
        if verdict and verdict != "correct": # treating incorrect and incomplete as wrong.
            error_reflections.append(reflection)

    return reflections


def main():
    

    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Test reflection diversity on trajectories")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file (e.g., iter1.evolved_kflow.jsonl)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index of questions to process (inclusive, default: 0)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1,
        help="End index of questions to process (exclusive, default: 1)"
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="base",
        choices=list(REFLECTION_SYSTEM_PROMPTS.keys()),
        help="Version of prompts to use (default: base)"
    )
    parser.add_argument(
        "--num_reflections",
        type=int,
        default=16,
        help="Number of reflections to generate for each trajectory (default: 16)"
    )

    args = parser.parse_args()

    # Get prompts based on version
    reflector_system_prompt = REFLECTION_SYSTEM_PROMPTS[args.prompt_version]
    reflector_user_template = REFLECTION_USER_TEMPLATES[args.prompt_version]
    reflector_tools_plain = REFLECTION_TOOLS[args.prompt_version]

    # Set environment variables for generate_reflections function
    os.environ["MAX_REFLECTIONS"] = str(args.num_reflections)

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    logger.info(f"Reading input file: {args.input_file}")
    logger.info(f"Processing questions from index {args.start_index} (inclusive) to {args.end_index} (exclusive)")
    trajectories_data = []

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            # Skip until we reach start_index
            if i < args.start_index:
                continue
            # Stop when we reach end_index
            if i >= args.end_index:
                break

            question_data = json.loads(line)
            # Extract the first history object (first iteration)
            if "history" in question_data and len(question_data["history"]) > 0:
                first_history = question_data["history"][0]
                trajectory = first_history["trajectory"]
                trajectories_data.append(trajectory)

    logger.info(f"Extracted {len(trajectories_data)} trajectories from input file")

    # Generate reflections for each trajectory
    results = []
    for idx, traj_data in enumerate(trajectories_data):
        logger.info(f"Processing trajectory {idx + 1}/{len(trajectories_data)}")

        reflections = generate_reflections(
            trajectory=traj_data,
            reflector_system_prompt=reflector_system_prompt,
            reflector_user_template=reflector_user_template,
            reflector_tools_plain=reflector_tools_plain,
        )

        results.append({
            "question": traj_data.get("question", ""),
            "answer": traj_data.get("answer", ""),
            "trajectory": traj_data,
            "reflections": reflections
        })

    # Write output file
    input_stem = input_path.stem  # filename without extension
    output_filename = f"{input_stem}.q{args.start_index}-{args.end_index}.nref{args.num_reflections}.jsonl"
    output_path = input_path.parent / output_filename

    logger.info(f"Writing output to: {output_path}")
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Done! Processed {len(results)} questions (indices {args.start_index}-{args.end_index}) with {args.num_reflections} reflections each")
    logger.info(f"Output written to: {output_path}")

if __name__ == "__main__":
    main()
