import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from evolve.reflector import Reflector
from evolve.llm import LLMClient
from evolve.evolver import Evolver
from evolve.generator import Generator
from evolve.curator import Curator
from react_agent import MultiTurnReactAgent
import time
import traceback


import math
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    parser.add_argument("--port", type=int, default=6001, help="vLLM server port")
    parser.add_argument(
        "--mode",
        type=str,
        default="evolve",
        choices=["baseline", "evolve", "evolve_kflow"],
        help="Mode to run: 'baseline' uses test_agent, 'evolve' uses evolver",
    )
    parser.add_argument(
        "--debug-size",
        type=int,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("MAX_ITERATIONS", "8")),
        help="Maximum evolution iterations for evolver (default: from MAX_ITERATIONS env or 8)",
    )
    args = parser.parse_args()

    wandb_run = wandb.init()
    print(f"WANDB_RUN_ID={wandb_run.id}", flush=True)

    print(args)
    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))

    model_dir = os.path.join(output_base, f"{model_name}")
    dataset_dir = os.path.join(model_dir, args.dataset, datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    # Prepare data
    # data_filepath = f"{args.dataset}"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_filepath = os.path.join(cur_dir, "eval_data", f"{args.dataset}.jsonl")
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if args.debug_size > 0:
        print(f">>>> DEBUG: using the first {args.debug_size} samples.")
        items = items[:args.debug_size]


    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_queries_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    # Define ports - using single port from command line argument
    planning_ports = [args.port]
    # Round-robin state
    planning_rr_idx = 0
    summary_rr_idx = 0
    # Sticky assignment per question
    question_to_ports = {}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                # Ensure sticky and balanced port assignment per unique question
                if question not in question_to_ports:
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1
                planning_port = question_to_ports[question]
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": planning_port,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")


    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            },
            'model_type': 'qwen_dashscope'
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            ####################
            function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
            ####################
            # DEBUG: limit to only search tool
            # function_list=["search"]
            ####################
        )


        #####################
        # Modify the job queue to use a controller (test_agent + reflector...etc) 
        llm_client = LLMClient(model_name=model, base_url=f"http://localhost:{args.port}/v1")
        reflector = Reflector(llm=llm_client)
        generator = Generator(task_agent=test_agent, model_name=model,)
        curator = Curator(llm=llm_client)

        evolver = Evolver(
            reflector=reflector,
            generator=generator,
            curator=curator,
        )
        #####################

        ################
        # DEBUG
        # res = evolver.evolve(tasks_to_run_all[0], max_iterations=2)
        # exit()
        ################

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            if args.mode == "baseline":
                run_func = lambda task: test_agent._run(task, model)
            elif args.mode == "evolve":
                run_func = lambda task: evolver.evolve(task, max_iterations=args.max_iterations)
            elif args.mode == "evolve_kflow":
                run_func = lambda task: evolver.evolve_kflow(task, max_iterations=args.max_iterations)
            else:
                raise NotImplementedError()

            future_to_task = {
                executor.submit(run_func, task): task for task in tasks_to_run_all
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    ###############################
                    if args.mode == "baseline":
                        result = future.result()
                        with write_locks[rollout_idx]:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    elif args.mode == "evolve":
                        result = future.result()
                        with write_locks[rollout_idx]:
                            with open(output_file.replace(".jsonl", ".evolved.jsonl"), "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    elif args.mode == "evolve_kflow":
                        result = future.result()
                        with write_locks[rollout_idx]:
                            with open(output_file.replace(".jsonl", ".evolved_kflow.jsonl"), "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        raise NotImplementedError()
                    ###############################
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]"
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file.replace(".jsonl", ".timeout.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc} - {"".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    with write_locks[rollout_idx]:
                        with open(output_file.replace(".jsonl", ".err.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

                ##################################
                if args.mode == "baseline":
                    wandb.save(output_file)
                elif args.mode == "evolve":
                    wandb.save(output_file.replace(".jsonl", ".evolved.jsonl"))
                elif args.mode == "evolve_kflow":
                    wandb.save(output_file.replace(".jsonl", ".evolved_kflow.jsonl"))
                else:
                    raise NotImplementedError()
                ##################################

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
    
    # Add a sentiniel doc 
    with open(os.path.join(dataset_dir, "._ok"), 'w') as f:
        f.write("All rollouts completed. Run succeeded.")
