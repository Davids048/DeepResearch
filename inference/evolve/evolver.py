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
from evolve.reflector_prompt import COMPRESSION_SYSTEM_PROMPT, COMPRESSION_TOOLS_PLAIN, COMPRESSION_USER_TEMPLATE, REFLECTION_TOOLS_KFLOW_PLAIN, REFLECTOR_SYSTEM_PROMPT, REFLECTOR_TEMPLATE_KFLOW
from evolve.utils import get_glm_openai_client, get_glm_tokenizer
from prompt_builder import tools_plain2openai
from parse_tools_utils import parse_model_response
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
            
            # 1. GENERATE: Task agent generates a trajectory
            # Phase 3: Pass previous reflection to guide next attempt
            additional_instructions = """\n\nFirst read the reviews on previous attempts, think about how to use the information to help you with solving the question. Then solve this problem."""
            trajectory = self.generator.generate(
                task=task,
                knowledge_history=knowledge_history,
                additional_instructions=additional_instructions if knowledge_history else None,
            )


            logger.debug(f"Task agent finished iteration {iteration}")
            # 2. REFLECT: Evaluate the attempt
            # Create reflector propmt
            messages = trajectory["messages"]
            prediction = trajectory["prediction"] 

            reflector_system_prompt = """
You are an expert evaluator specializing in analyzing and reflecting on the performance of AI assistants in multi-step reasoning and search tasks.

Your goal is to identify errors, inefficiencies, and opportunities for improvement in the assistant's reasoning and decision-making process.

## Hard Rules
- Do **not** attempt to solve the problem yourself or infer the correct answer.  
- Base your reflection solely on the assistant's reasoning and action trajectory.  
- Do **not** use external tools or perform additional searches during analysis.  
- Your evaluation must remain grounded in the assistant’s own process and content.  

## Output Requirement
After completing your analysis, you **must** use the `output_json_reflection` tool to produce a JSON object summarizing your reflection.  
Before returning, verify that the **only** tool you invoked is `output_json_reflection`.
"""

            reflector_user_template = """
You will be given the following materials:
- **trace**: the user query and the assistant’s reasoning/action history.  
- **prediction**: the assistant’s final output.  

## Instructions
1. Carefully read through the assistant’s full trajectory.  
2. Define **{num_rubrics} distinct rubrics** for evaluating the trajectory (e.g., reasoning accuracy, completeness, efficiency, self-consistency, tool use, etc.).  
3. For each rubric, assess the assistant’s performance.  
   - Identify any flaws, inefficiencies, or reasoning gaps.  
   - Highlight critical mistakes and propose potential improvements.
4. Output exactly {num_rubrics} rubric-based reviews.

## Data
### Assistant's Message Trajectory
===============================================================================
{messages}
===============================================================================
**End of Trajectory**

### Final Assistant Prediction
{prediction}
"""

            reflector_tools_plain = [{
                "name": "output_json_reflection",
                "description": "Produce a JSON summary reflecting on the assistant’s performance across defined rubrics that strictly follows this schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "review": {
                            "type": "array",
                            "description": "A list of strings. Each string is a rubric-based reflections assessing the assistant’s performance.",
                            "items": {
                                "type": "string",
                                "description": "One reflection entry formatted as '**Rubric Name**: Evaluation and suggestions for improvement.'"
                            }
                        }
                    },
                    "required": ["review"],
                    "additionalProperties": False
                }
            }]

            num_reflections = int(os.getenv("MAX_REFLECTIONS", 16))           
            reflector_user_prompt = reflector_user_template.format(
                messages = messages,
                prediction = prediction,
                num_rubrics = num_reflections,
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
                        parsed_response = parse_model_response(response, reflector_tools_plain)
                        reasoning_content = parsed_response.get("reasoning_content", "")
                        tool_calls = parsed_response.get("tool_calls", [])
                        data = tool_calls[0]["arguments"]
                        data["reasoning_content"] = reasoning_content
                    except Exception as e:
                        logger.error(f"Unexpected error parsing reflector response, using fallback output. Error: {e}. Raw response:{parsed_response}")
                        # Continue to next trial
                        continue

                    # Ensure the data is in the right format (review must be a list, not a string)
                    if "review" in data:
                        review = data["review"]
                        # If review is not a list, regenerate
                        if not isinstance(review, list):
                            logger.warning(f"Review has wrong type: {type(review)}, regenerating (trial {i+1}/5)")
                            data = None
                            continue
                        # Valid format - break out of retry loop
                        logger.debug(f"Valid review format found on trial {i+1}")
                        break
                    else:
                        # If review field is missing, regenerate
                        logger.warning(f"Review field missing from reflector response, regenerating (trial {i+1}/5)")
                        data = None
                        continue

                # If all trials failed, return fallback with empty review list
                if data is None:
                    logger.error(f"All trials failed to generate valid review format, using fallback")
                    data = {
                        "error": "All trials failed to generate valid review",
                        "review": []
                    }

                return data 
            reflections = []
            num_reflections = 1  ### Since we are genearting num_reflections rubrics in one generation, here we set num_reflections to 1. 
            num_reflection_workers = int(os.getenv("MAX_REFLECTION_WORKERS", 16))           
            with ThreadPoolExecutor(max_workers = num_reflection_workers) as executor:
                futures = [executor.submit(_generate_single_reflection) for _ in range(num_reflections)]
                for future in as_completed(futures):
                    try:
                        reflections.append(future.result())
                    except Exception as e:
                        logger.error(f"Reflection future failed: {e}.")
                        reflections.append({"error": "Reflection failed"})

            
            review_items = reflections[0].get("review", []) if reflections else []
            # Ensure there are only num_reflections rubrics in the generated reviews list
            if isinstance(review_items, list) and len(review_items) > num_reflections:
                logger.warning(f"Generated {len(review_items)} rubrics, but expected {num_reflections}. Randomly selecting {num_reflections} rubrics.")
                review_items = random.sample(review_items, num_reflections)

            # join into a single readable evaluation text
            if isinstance(review_items, list):
                summarized_reflection = "\n\n".join(review_items)
            elif isinstance(review_items, str):
                summarized_reflection = review_items 
                logger.error(f"review items is string. Using fault tolerant formatting...")
            else:
                logger.error(f"Unexpected review items type: {type(review_items)}")
            knowledge_history.append(summarized_reflection)
            # Aggregate the results - pickout the ones where reflector judge the trace as wrong. 
            # error_reflections = []
            # for reflection in reflections:
            #     verdict = reflection.get("correctness_judgement", "")
            #     if verdict and verdict != "correct": # treating incorrect and incomplete as wrong.
            #         error_reflections.append(reflection)

            # # Compress Reflections into 1 report. 
            # compression_system_prompt = COMPRESSION_SYSTEM_PROMPT
            # compression_user_template = COMPRESSION_USER_TEMPLATE
            # compression_tools_plain = COMPRESSION_TOOLS_PLAIN

            # summarized_reflection = "N/A"
            # if error_reflections:
            #     question = trajectory["question"]
            #     prediction = trajectory["prediction"]
            #     reflections_text = ""
            #     for i, reflection in enumerate(error_reflections):
            #         reflections_text += (
            #             f"# Review {i}:\n"
            #             f"## Assistant Trajectory Summary: {reflection.get('trajectory_summary', '')}\n\n"
            #             f"## Error Report: {reflection.get('error_report', '')}\n\n"
            #             f"## Rubric Based Evaluation: {reflection.get('rubric_based_evaluation', '')}\n\n"
            #         )

            #     # format compression prompt 
            #     compression_prompt = compression_user_template.format(
            #         question = question,
            #         prediction = prediction,
            #         reflections = reflections_text,
            #     )
            #     tokenizer = get_glm_tokenizer()
            #     prompt = tokenizer.apply_chat_template(
            #         [
            #             {"role": "system", "content": compression_system_prompt},
            #             {"role": "user", "content": compression_prompt},
            #         ],
            #         tools = tools_plain2openai(compression_tools_plain),
            #         tokenize=False,
            #         enable_thinking=True,
            #         add_generation_prompt=True,
            #     )
            #     logger.debug(f"compression prompt: {prompt}")
            #     client = get_glm_openai_client()
            #     response = client.completions.create(
            #         model="zai-org/GLM-4.6",
            #         prompt = prompt,
            #         **DEFAULT_COMPLETION_CONFIG,
            #     )
            #     response = response.choices[0].text
            #     response = "<think>" + response
            #     logger.debug(f"compressed report: {response}")

            #     try:
            #        # GLM is using pure text handling.
            #        parsed_response = parse_model_response(response, COMPRESSION_TOOLS_PLAIN)
            #        reasoning_content = parsed_response.get("reasoning_content", "")
            #        tool_calls = parsed_response.get("tool_calls", [])
            #        data = tool_calls[0]["arguments"]
            #        data["reasoning_content"] = reasoning_content
            #        summarized_reflection = data
            #     except Exception as e:
            #         logger.error(f"Unexpected error parsing reflector response, using fallback output. Error: {e}")
            #         data = {
            #             "error": "Reflector encountered unexpected error",
            #         }
            #     knowledge_history.append(summarized_reflection)


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

