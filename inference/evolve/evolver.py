from dataclasses import asdict
import os
from typing import Optional
from evolve.generator import Generator
from evolve.reflector import Reflector, ReflectorOutput
from evolve.curator import Curator
from evolve.playbook import Playbook
from logger import setup_logging

logger = setup_logging(name=__name__, level=5)

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
            "rounds": last_iteration["trajectory"]["rounds"],
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

