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
        self.playbook = playbook or Playbook()


    def evolve(
        self, 
        task:dict,
    ):
        # task agent generates a trajectory 
        trajectory = self.generator.generate(
            task=task,
            playbook=self.playbook,
            reflection=None,
            debug=True,
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

