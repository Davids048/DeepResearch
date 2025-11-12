from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import os
import random
from typing import Optional
from evolve.generator import Generator
from evolve.reflector import Reflector, ReflectorOutput
from evolve.curator import Curator
from evolve.playbook import Playbook
from evolve.llm import DEFAULT_COMPLETION_CONFIG
from evolve.utils import get_glm_openai_client, get_glm_tokenizer
from prompt_builder import tools_plain2openai
from parse_tools_utils import parse_model_response, parse_model_response_json
from logger import setup_logging

logger = setup_logging(name=__name__, level=20)

class Evolver:
    """A controller to manage the interaction between task agent, reflector, curators, memory (and other components)."""

    def __init__(
        self, 
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        playbook: Optional[Playbook] = None
    ):
        self.reflector = reflector
        self.generator = generator
        self.curator = curator
        
        #############################
        # DEBUG MODE: Create an initial playbook. 
        # self.playbook = self._create_mock_playbook()
        # logger.debug(f"DEBUG mode: Created mock playbook with {len(self.playbook.bullets())} bullets")
        #############################
        self.playbook = playbook or Playbook()
        #############################


    def evolve(
        self,
        task: dict,
        max_iterations: int = 8,
        stop_on_correct: bool = True,
    ):
        """Evolve the agent through iterative refinement on a single task.

        Args:
            task: Task to solve (must contain 'question' and 'answer')
            max_iterations: Maximum evolution iterations (default: 1 for backward compatibility)
            stop_on_correct: Stop iterating if correct answer achieved (default: True for efficiency)

        Returns:
            Dict containing:
                - trajectory: Last iteration's trajectory (backward compatible)
                - reflection: Last iteration's reflection (backward compatible)
                - curation: Last iteration's curation (backward compatible)
                - history: List of all iterations (Phase 1)
                - iterations_used: Number of iterations executed (Phase 1)
                - final_correctness: Last iteration's correctness judgement (Phase 2)
                - achieved_correct: Whether correct answer was achieved (Phase 2)
        """
        history = []
        previous_reflection_text = None  # Phase 3: Track reflection from previous iteration
        ##############
        # DEBUG 
        # previous_reflection_text = self._create_mock_reflection()
        ##############

        logger.info(f"Starting evolution with max_iterations={max_iterations}, stop_on_correct={stop_on_correct}")

        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Evolution Iteration {iteration}/{max_iterations} ===")

            # 1. GENERATE: Task agent generates a trajectory
            # Phase 3: Pass previous reflection to guide next attempt
            trajectory = self.generator.generate(
                task=task,
                playbook=self.playbook,
                reflection=previous_reflection_text,
            )

            # trajectory = {
            #     "question": task.get('item', {}).get('question', 'Mock question'),
            #     "answer": task.get('item', {}).get('answer', 'Mock answer'),
            #     "messages": [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": task.get('item', {}).get('question', 'Mock question')},
            #         {"role": "assistant", "reasoning_content": "Let me think about this question.", "tool_calls": []},
            #         {"role": "assistant", "reasoning_content": "Based on my analysis, the answer is: Mock answer", "tool_calls": [{"name": "finish", "arguments": {}}]},
            #     ],
            #     "prediction": "Mock answer",
            #     "termination": "answer",
            #     "rounds": 2,
            # }

            logger.debug(f"Task agent finished iteration {iteration}")

            # 2. REFLECT: Evaluate the attempt
            reflector_output = self.reflector.reflect(
                trajectory=trajectory,
                playbook=self.playbook,
            )

            # 3. TAG: Apply bullet tags based on performance
            self._apply_bullet_tags(reflector_output)

            # 4. CHECK CORRECTNESS (Phase 2)
            correctness = reflector_output.correctness_judgement.lower().strip()
            is_correct = correctness == "correct"

            logger.info(f"Iteration {iteration} reflector judgement: {correctness}")

            # 5. CURATE: Update playbook
            try:
                question = task['item']['question']
            except:
                raw_msg = task['item']['messages'][1]["content"]
                question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

            curator_output = self.curator.curate(
                question_context=question,
                playbook=self.playbook,
                reflector_output=reflector_output,
            )

            self.playbook.apply_delta(curator_output.delta)

            logger.debug(f"Playbook after iteration {iteration}: {len(self.playbook.bullets())} bullets")

            # 6. RECORD: Save iteration result
            iteration_result = {
                "iteration": iteration,
                "trajectory": trajectory,
                "reflection_output": asdict(reflector_output),
                "curation": asdict(curator_output),
                "reflector_judgement": correctness,
            }
            history.append(iteration_result)

            logger.info(f"Iteration {iteration} completed")

            # 7. PREPARE: Format reflection for next iteration (Phase 3)
            if iteration < max_iterations:
                previous_reflection_text = self._format_reflection_for_next_iteration(reflector_output)
                logger.debug(f"Prepared reflection for next iteration:\n{previous_reflection_text}")

            # 8. EARLY STOPPING (Phase 2)
            if is_correct and stop_on_correct:
                logger.info(f"✅ Correct answer achieved at iteration {iteration}/{max_iterations} - stopping early")
                break

        logger.info(f"Evolution completed: {len(history)} iterations executed")

        # Return last iteration for backward compatibility + full history
        last_iteration = history[-1]
        return {
            "question": last_iteration["trajectory"]["question"],
            "answer": last_iteration["trajectory"]["answer"],
            "iterations_used": len(history),
            "final_prediction": last_iteration["trajectory"]["prediction"],
            "final_termination": last_iteration["trajectory"]["termination"],
            "final_reflector_judgement": last_iteration["reflector_judgement"],
            "final_rounds": last_iteration["trajectory"]["rounds"],
            "history": history,
        }


    def evolve_kflow(
        self,
        task: dict,
        max_iterations: int = 8,
        stop_on_correct: bool = True,
    ):
        """Evolve the agent through iterative refinement on a single task.

        Args:
            task: Task to solve (must contain 'question' and 'answer')
            max_iterations: Maximum evolution iterations (default: 1 for backward compatibility)
            stop_on_correct: Stop iterating if correct answer achieved (default: True for efficiency)

        Returns:
            Dict containing:
                - trajectory: Last iteration's trajectory (backward compatible)
                - reflection: Last iteration's reflection (backward compatible)
                - curation: Last iteration's curation (backward compatible)
                - history: List of all iterations (Phase 1)
                - iterations_used: Number of iterations executed (Phase 1)
                - final_correctness: Last iteration's correctness judgement (Phase 2)
                - achieved_correct: Whether correct answer was achieved (Phase 2)
        """
        history = []
        logger.info(f"Starting evolution with max_iterations={max_iterations}, stop_on_correct={stop_on_correct}")
        
        knowledge_history = [] # a list of all the previous summarized reports.
        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Evolution Iteration {iteration}/{max_iterations} ===")
            trajectory = None
            if iteration > 1:
            # 1. GENERATE: Task agent generates a trajectory
                additional_instructions = """\n\nRead the reviews based on previous attempts first, then solve the problem leveraging each relevant proposed adjustments."""
                trajectory = self.generator.generate(
                    task=task,
                    knowledge_history=knowledge_history,
                    additional_instructions=additional_instructions if knowledge_history else None,
                )
            else:
                ##########################
                # # DEBUG: USE AN INPUT FILE AND GET the trajectory with the corresponding question.
                import json
                input_file = "../inference/output/GLM-4.6/browsecomp/20251111-010304/iter1.jsonl"
                # Extract the question from task
                try:
                    task_question = task['item']['question']
                except:
                    raw_msg = task['item']['messages'][1]["content"]
                    task_question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

                # Load the file and find matching trajectory
                with open(input_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get('question') == task_question:
                            trajectory = data
                            logger.info(f"DEBUG: Loaded trajectory from {input_file} for question: {task_question[:50]}...")
                            break
                ##########################
            assert trajectory is not None

            logger.debug(f"Task agent finished iteration {iteration}")
            # 2. REFLECT: Evaluate the attempt
            # Create reflector propmt
            messages = trajectory["messages"]
            prediction = trajectory["prediction"] 
            
            # If messages have more than 120 rounds, drop the rounds with role = tool
            if len(messages) > 120:
                logger.warning(f"Messages have {len(messages)} rounds, filtering out tool messages to reduce size")
                messages = [msg for msg in messages if msg.get("role") != "tool"]
                logger.info(f"After filtering, messages have {len(messages)} rounds") 

            reflector_system_prompt = """
You are an expert evaluator specializing in analyzing and reflecting on the performance of AI assistants in multi-step reasoning and search tasks.

Your goal is to identify errors, inefficiencies, and opportunities for improvement in the assistant's reasoning and decision-making process.

## Hard Rules
- Base your reflection solely on the assistant's reasoning and action trajectory.  
- Do **not** use external tools or perform additional searches during analysis.  
- Your evaluation must remain grounded in the assistant’s own process and content.  
- Do not include any concrete clues or details from the previous attempt other than those mentioned in the original question.
- Zero-leak policy: if a detail appears in the trajectory but not verbatim in the original question, it is forbidden.

"""

            reflector_user_template = """
You will be given the following materials:
- **trace**: the user query and the assistant’s reasoning/action history.  
- **prediction**: the assistant’s final output.  

## Perform the following steps:
1. First think carefully how you would have approached the question, including assumptions, search strategies, pivots. 
2. Carefully read through the assistant’s full trajectory.  Summarize key assumptions, strategies, and explored search space.
3. Compare what is different from what you would have done.
4. Based on your comparison, write some proposed_adjustments. Write like an inner monologue planning a pivot for the next attempt. Use the format: "Let me think outside the box, what if...?"
5. Remove every detail, name, location, time, unless it appeared in the original question.


## Hard rules for proposed_adjustments:
- Focus on how to rethink the assumptions, reasoning, or search strategy, not on reusing or referring to specific names, facts, or partial answers from the trajectory. 
- Do not include any concrete entities or details from the previous attempt.
- Focus only on rethinking assumptions, reasoning, or search strategy. 
- Do not reuse or refer to specific names, facts, numbers, or partial answers from the trajectory unless they appear verbatim in the original question.
- If any forbiddent examples, names, locations, time remains, remove it or replace with a placeholder.

## Output Requirement
Output a json object wrapped in ``` blocks including the following fields: 
- proposed_adjustments: Your proposed pivots. 

## Example Output Format:
<Your reasoning> 
```
{{
    "proposed_adjustments": <your proposed adjustments>,
}}
```


## Data
### Assistant's Message Trajectory
===============================================================================
{messages}
===============================================================================
**End of Trajectory**

### Final Assistant Prediction
{prediction}
"""

            num_reflections = int(os.getenv("MAX_REFLECTIONS", 16))
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
                # tools = tools_plain2openai(reflector_tools_plain),
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
            client = get_glm_openai_client()
            def _generate_single_reflection():
                data = None
                for i in range(5): # perform 5 trials, ensure the output format
                    response = client.completions.create(
                        model="zai-org/GLM-4.6",
                        prompt = prompt,
                        # Use following params for more variety.
                        temperature=1.0,
                        top_p = 0.95,
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
                        # parsed_response = parse_model_response(response, reflector_tools_plain)
                        # Use Parse Json.
                        parsed_response = parse_model_response_json(response)
                        reasoning_content = parsed_response.get("reasoning_content", "")
                        tool_calls = parsed_response.get("tool_calls", [])
                        data = tool_calls[0]["arguments"]
                        data["reasoning_content"] = reasoning_content
                    except Exception as e:
                        logger.error(f"Unexpected error parsing reflector response, using fallback output. Error: {e}. Raw response:{parsed_response}")
                        # Continue to next trial
                        continue

                    # Ensure the data is in the right format
                    required_fields = ["proposed_adjustments"]
                    if all(field in data for field in required_fields):
                        proposed_adjustments = data["proposed_adjustments"]
                        # Validate that all fields are strings
                        if not isinstance(proposed_adjustments, str):
                            logger.warning(
                                f"Invalid field types: "
                                f"proposed_adjustments={type(proposed_adjustments)}, "
                                f"regenerating (trial {i+1}/5)"
                            )
                            data = None
                            continue
                        # Valid format - break out of retry loop
                        logger.debug(f"Valid reflection format found on trial {i+1}")
                        break
                    else:
                        # If required fields are missing, regenerate
                        missing_fields = [field for field in required_fields if field not in data]
                        logger.warning(f"Required fields missing from reflector response: {missing_fields}, regenerating (trial {i+1}/5)")
                        data = None
                        continue

                # If all trials failed, return fallback
                if data is None:
                    logger.error(f"All trials failed to generate valid reflection format, using fallback")
                    data = {
                        "error": "All trials failed to generate valid reflection",
                        "proposed_adjustments": "Unable to generate reflection after multiple attempts."
                    }

                return data 
            reflections = []
            num_reflection_workers = int(os.getenv("MAX_REFLECTION_WORKERS", 16))           
            with ThreadPoolExecutor(max_workers = num_reflection_workers) as executor:
                futures = [executor.submit(_generate_single_reflection) for _ in range(num_reflections)]
                for future in as_completed(futures):
                    try:
                        reflections.append(future.result())
                    except Exception as e:
                        logger.error(f"Reflection future failed: {e}.")
                        reflections.append({"error": "Reflection failed"})
            
            ################# COMPRESSION #################
            # Aggregate all reflections (no filtering based on correctness)
            error_reflections = reflections

            # Compress Reflections into 1 report. 
            compression_system_prompt = """
You are an expert summarizer. Your task is to read multiple reviewer reports, deduplicate them, and remove any sensitive information.

# Output format:
- After your analysis, you MUST use the 'output_json_summrized_reflction' tool to produce a json object of your reflection
- Before return, double check that the only tool you called is 'output_json_summrized_reflction'
"""

            compression_user_template = """
# General Context
You will be provided with the following materials:
- The original question given to the assistant.
- Multiple reviewer reports on an assistant's trajectory. Each report will have some proposed adjustments. 

# Key Instructions:
- Remove any details, examples, names, times, locations that did not appear in the original question. 
- Preserve the original reviewer's phrasing when possible.
- You may drop or merge semantically equivalent items, even if they differ slightly in phrasing, punctuation, or word choice.
- Perform semantic deduplication only. If two or more adjustments express the same actionable idea, keep exactly one and delete all others.

# Note: 
- Treat two adjustments as duplicates if they differ only in surface form (e.g., synonyms, tense, or formatting) but share the same intent or suggestion.
- Treat items as duplicates even if they use different wording, as long as they point the assistant to the same change.
- The final list must contain only unique adjustments. No adjustment should appear more than once.

# Hard rules
- Do not elevate to higher-level summaries. Do not introduce titles, headings, or new structure.
- No details, examples, names, times, locations that did not appear in the original question should appear in the final output. 
- If any of the above exist, remove them from the final output. 

Below are the information needed for summarization: 
### Original question 
{question} 

### Reviewer reports 
{reflections}
"""

            compression_tools_plain = [{
                "name": "output_json_summrized_reflction",
                "description": "Output a json object of the reflection on the task agent's trajectory. When producing summarized_proposed_adjustments, include only the deduplicated list.",
                "parameters": {
                    'type': 'object',
                    'properties': {
                        "summarized_proposed_adjustments": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": (
                                "An array of deduplicated proposed adjustments, each formatted as: **Reviewer**: <original proposed adjustment>"
                            )
                        }
                    },
                    'required': [],
                    'additionalProperties': False,
                }
            }]

            summarized_reflection = "N/A"
            if error_reflections:
                question = trajectory["question"]
                prediction = trajectory["prediction"]
                reflections_text = ""
                for i, reflection in enumerate(error_reflections):
                    reflections_text += (
                        f"# Review {i}:\n"
                        f"## Proposed adjustments: {reflection.get('proposed_adjustments', '')}\n\n"
                    )

                # format compression prompt 
                compression_prompt = compression_user_template.format(
                    question = question,
                    reflections = reflections_text,
                )
                tokenizer = get_glm_tokenizer()
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": compression_system_prompt},
                        {"role": "user", "content": compression_prompt},
                    ],
                    tools = tools_plain2openai(compression_tools_plain),
                    tokenize=False,
                    enable_thinking=True,
                    add_generation_prompt=True,
                )
                logger.debug(f"compression prompt: {prompt}")
                client = get_glm_openai_client()

                # Add retrial logic for compression (similar to reflection)
                data = None
                for i in range(5):  # perform 5 trials, ensure the output format
                    response = client.completions.create(
                        model="zai-org/GLM-4.6",
                        prompt = prompt,
                        **DEFAULT_COMPLETION_CONFIG,
                    )
                    response = response.choices[0].text
                    response = "<think>" + response
                    logger.debug(f">>>>>>>>>> compression response (trial {i+1}/5): {response}.")

                    try:
                        # GLM is using pure text handling.
                        parsed_response = parse_model_response(response, compression_tools_plain)
                        reasoning_content = parsed_response.get("reasoning_content", "")
                        tool_calls = parsed_response.get("tool_calls", [])
                        data = tool_calls[0]["arguments"]
                        data["reasoning_content"] = reasoning_content
                    except Exception as e:
                        logger.error(f"Unexpected error parsing compression response, using fallback output. Error: {e}. Raw response:{parsed_response}")
                        # Continue to next trial
                        continue

                    # Ensure the data is in the right format
                    if "summarized_proposed_adjustments" in data:
                        summarized_proposed_adjustments = data["summarized_proposed_adjustments"]
                        # Validate that field is a list
                        if not isinstance(summarized_proposed_adjustments, list):
                            logger.warning(f"Invalid field type: summarized_proposed_adjustments={type(summarized_proposed_adjustments)}, regenerating (trial {i+1}/5)")
                            data = None
                            continue
                        # Validate all items in the list are strings
                        if not all(isinstance(item, str) for item in summarized_proposed_adjustments):
                            logger.warning(f"Invalid list items: not all items in summarized_proposed_adjustments are strings, regenerating (trial {i+1}/5)")
                            data = None
                            continue
                        # Valid format - break out of retry loop
                        logger.debug(f"Valid compression format found on trial {i+1}")
                        break
                    else:
                        # If required field is missing, regenerate
                        logger.warning(f"Required field 'summarized_proposed_adjustments' missing from compression response, regenerating (trial {i+1}/5)")
                        data = None
                        continue

                # If all trials failed, return fallback
                if data is None:
                    logger.error(f"All trials failed to generate valid compression format, using fallback")
                    data = {
                        "error": "Compression encountered unexpected error after multiple attempts",
                        "summarized_proposed_adjustments": ["Unable to compress reflections after multiple attempts."]
                    }

                summarized_reflection = data
                knowledge_history.append(summarized_reflection)


            iteration_result = {
                "iteration": iteration,
                "trajectory": trajectory,
                "reflection_output": reflections,
                "summarized_reflection": summarized_reflection,
            }
            history.append(iteration_result)

            logger.info(f"Iteration {iteration} completed")


        logger.info(f"Evolution completed: {len(history)} iterations executed")

        # Return last iteration for backward compatibility + full history
        last_iteration = history[-1]
        return {
            "question": last_iteration["trajectory"]["question"],
            "answer": last_iteration["trajectory"]["answer"],
            "iterations_used": len(history),
            "final_prediction": last_iteration["trajectory"]["prediction"],
            "final_termination": last_iteration["trajectory"]["termination"],
            "final_rounds": last_iteration["trajectory"]["rounds"],
            "history": history,
        }




    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue

    def _format_reflection_for_next_iteration(self, reflection: ReflectorOutput) -> str:
        """Format reflection as context for next generation attempt.

        This follows the Phase 3 design: pass previous iteration's insights
        to guide the next attempt at solving the task.

        Uses dataclass fields dynamically to be resistant to ReflectorOutput changes.

        Args:
            reflection: ReflectorOutput from previous iteration

        Returns:
            Formatted string to pass to generator's reflection parameter
        """
        from dataclasses import fields

        sections = []
        sections.append("## Previous Attempt Analysis")
        sections.append("")

        # Get all fields from the dataclass
        for field in fields(reflection):
            # Skip internal/raw fields and bullet_tags (handled separately)
            if field.name in ["raw", "bullet_tags"]:
                continue

            value = getattr(reflection, field.name)

            # Skip empty or None values
            if not value or (isinstance(value, str) and value.lower().strip() in ["none", "n/a", ""]):
                continue

            # Use field name as label
            label = field.name.replace("_", " ").title()

            sections.append(f"**{label}:** {value}")
            sections.append("")

        # Add actionable guidance based on correctness
        correctness = reflection.correctness_judgement.lower().strip()
        if correctness != "correct":
            sections.append("**Action:** Apply these insights to improve your next attempt and avoid repeating the same mistakes.")
        else:
            sections.append("**Action:** The previous attempt was successful. Use these insights as reference.")

        return "\n".join(sections)

    @staticmethod
    def _create_mock_reflection() -> str:
        """Create a mock reflection for testing purposes.

        This creates a fake reflection stating the agent succeeded, to test
        whether the agent properly receives and uses reflection context.

        Returns:
            Formatted reflection text similar to _format_reflection_for_next_iteration output
        """
        return """## Previous Attempt Analysis

**Error Identification:** None

**Root Cause Analysis:** None

**Correct Approach:** The agent successfully completed the task with accurate research and proper tool usage.

**Key Insight:** Mock reflection for testing - if you see this message, the reflection context passing is working correctly.

**Correctness Judgement:** correct

**Action:** The previous attempt was successful. Use these insights as reference."""


    @staticmethod
    def _create_mock_playbook() -> Playbook:
        """Create a mock playbook for debugging purposes.

        Returns:
            Playbook instance with sample strategic guidance
        """
        playbook = Playbook()

        # Add mock bullet based on the debug example
        playbook.add_bullet(
            section="strategies_and_hard_rules",
            content=(
                "Prioritize factual accuracy by acknowledging uncertainty and avoiding fabrication. "
                "When search tools fail or information is insufficient, explicitly state: "
                "'I cannot verify [specific detail] with current constraints.' "
                "Never invent names, dates, or biographical details. "
                "Only state logical constraints derived from verified facts "
                "(e.g., 'Without verified author identity, exact years cannot be determined, "
                "but probation officer role must have ended before 2018 when lecturing began')."
            ),
            bullet_id="strategies_and_hard_rules-00001"
        )
        playbook._next_id=1

        return playbook

