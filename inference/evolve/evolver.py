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

        logger.info(f"Starting evolution with max_iterations={max_iterations}, stop_on_correct={stop_on_correct}")

        for iteration in range(1, max_iterations + 1):
            logger.info(f"=== Evolution Iteration {iteration}/{max_iterations} ===")

            # 1. GENERATE: Task agent generates a trajectory
            trajectory = self.generator.generate(
                task=task,
                playbook=self.playbook,
                reflection=None,  # Phase 3 will pass previous reflection here
            )

            logger.debug(f"Task agent finished iteration {iteration}")

            # 2. REFLECT: Evaluate the attempt
            reflector_output, reflection_summary = self.reflector.reflect(
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
            curator_output = self.curator.curate(
                question_context=task.get("question", ""),
                playbook=self.playbook,
                reflector_output=reflector_output,
            )

            self.playbook.apply_delta(curator_output.delta)

            logger.debug(f"Playbook after iteration {iteration}: {len(self.playbook.bullets())} bullets")

            # 6. RECORD: Save iteration result
            iteration_result = {
                "iteration": iteration,
                "trajectory": trajectory,
                "reflection_output": reflector_output,
                "reflection_summary": reflection_summary,
                "curation": curator_output,
                "correctness": correctness,
            }
            history.append(iteration_result)

            logger.info(f"Iteration {iteration} completed")

            # 7. EARLY STOPPING (Phase 2)
            if is_correct and stop_on_correct:
                logger.info(f"✅ Correct answer achieved at iteration {iteration}/{max_iterations} - stopping early")
                break

        logger.info(f"Evolution completed: {len(history)} iterations executed")

        # Return last iteration for backward compatibility + full history
        last_iteration = history[-1]
        return {
            # Backward compatible fields (last iteration)
            "trajectory": last_iteration["trajectory"],
            "reflection": last_iteration["reflection_summary"],
            "curation": last_iteration["curation"],
            # Phase 1 fields
            "history": history,
            "iterations_used": len(history),
            # Phase 2 fields
            "final_correctness": last_iteration["correctness"],
            "achieved_correct": last_iteration["correctness"] == "correct",
        }


    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue

