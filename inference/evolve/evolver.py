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
        self.playbook = self._create_mock_playbook()
        logger.debug(f"DEBUG mode: Created mock playbook with {len(self.playbook.bullets())} bullets")
        #############################
        # self.playbook = playbook or Playbook()
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
        task:dict,
    ):
        # task agent generates a trajectory 
        trajectory = self.generator.generate(
            task=task,
            playbook=self.playbook,
            reflection=None,
        )

        logger.debug(f"Task agent finished....") 

        reflector_output, reflection_summary = self.reflector.reflect(
            trajectory=trajectory,
            playbook=self.playbook,
        )

        self._apply_bullet_tags(reflector_output)

        curator_output = self.curator.curate(
            question_context=task.get("question", ""),
            playbook=self.playbook,
            reflector_output=reflector_output,
        )

        self.playbook.apply_delta(curator_output.delta)

        logger.debug(f"playbook after 1 evolve:{self.playbook.as_prompt()}.")

        return {
            "trajectory": trajectory,
            "reflection": reflection_summary,
            "curation": curator_output,
        }


    def _apply_bullet_tags(self, reflection: ReflectorOutput) -> None:
        for tag in reflection.bullet_tags:
            try:
                self.playbook.tag_bullet(tag.id, tag.tag)
            except ValueError:
                continue

