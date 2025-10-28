import json
from dataclasses import dataclass
import os
from typing import Dict, List, Any, Optional
from evolve.llm import LLMClient
from evolve.generator_prompt import GENERATOR_PROMPT
from react_agent import MultiTurnReactAgent
from evolve.playbook import Playbook
from prompt import SYSTEM_PROMPT

from logger import setup_logging 
logger = setup_logging(name=__name__, level=5)

# @dataclass
# class GeneratorOutput:
#     reasoning: str
#     final_answer: str
#     bullet_ids: List[str]
#     raw: Dict[str, Any]
#

class Generator:
    """Produces trajectories using the current playbook."""

    def __init__(
        self,
        task_agent: MultiTurnReactAgent,
        model_name: str,
        prompt_template: str = GENERATOR_PROMPT,
    ) -> None:
        self.task_agent = task_agent
        self.model_name = model_name
        self.prompt_template = prompt_template

    def _build_system_prompt_with_playbook(
        self,
        playbook: Optional[Playbook] = None,
    ) -> str:
        """Build enhanced system prompt with playbook guidance and reflection insights.

        Args:
            playbook: Playbook containing strategic guidance
            reflection: Recent reflection insights to incorporate

        Returns:
            Enhanced system prompt string
        """
        base_prompt = SYSTEM_PROMPT
        sections = []

        # Add playbook section if available
        if playbook:
            playbook_text = playbook.as_prompt()
            if playbook_text:
                sections.append("\n\n# Strategic Playbook\n")
                sections.append("The following playbook contains proven strategies and common pitfalls. Apply relevant guidance during your research:\n")
                sections.append(f"\n{playbook_text}")

        return base_prompt + "".join(sections)

    def generate(
        self,
        task: dict,
        playbook: Playbook = None,
        reflection: Optional[str] = None,
    ):
        #####################################
        # DEBUG reflector/curator.
        # 
        # logger.debug("Bypass actual generation. Use a mock response ")
        # # 
        # trajectory_path = "/home/hal-jundas/agent/DeepResearch/inference/output/Qwen3-235B-A22B-Thinking-2507/browsecomp/20251024-184158/iter1.evolved.jsonl"
        # with open(trajectory_path, "r") as f:
        #     raw_data = f.readline()
        #     data = json.loads(raw_data)
        #     trajectory = {
        #         "question": data.get("question"),
        #         "answer": data.get("answer"),
        #         "messages": data.get("messages"),
        #         "prediction": data.get("prediction"),
        #         "termination": data.get("termination")
        #     }
        # return trajectory
        #####################################

        #####################################
        logger.debug(f"playbook: {playbook is not None}; reflection: {reflection is not None}.")
        # regular case [add new prompt to generate]
        # Build enhanced system prompt with playbook and reflection
        system_prompt = self._build_system_prompt_with_playbook(playbook)
        logger.debug(f"system prompt:\n{system_prompt}...")

        trajectory = self.task_agent._run(
            data=task,
            model=self.model_name,
            system_prompt=system_prompt,
        )
        return trajectory
        #####################################


