import json
from dataclasses import dataclass
import os
from typing import Dict, List, Any, Optional
from evolve.llm import LLMClient
from evolve.generator_prompt import GENERATOR_PROMPT
from react_agent import MultiTurnReactAgent
from evolve.playbook import Playbook
from prompt_builder import build_system_prompt

from logger import setup_logging
logger = setup_logging(name=__name__, level=5)


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


    def generate(
        self,
        task: dict,
        playbook: Playbook = None,
        reflection: Optional[str] = None,
    ):
        logger.debug(f"playbook: {playbook is not None}; reflection: {reflection is not None}.")

        # Build system prompt using prompt builder
        system_prompt = build_system_prompt(
            model_name=self.model_name,
            playbook=playbook,
            reflection=reflection,
        )
        logger.debug(f"system prompt:\n{system_prompt}...")

        trajectory = self.task_agent._run(
            data=task,
            model=self.model_name,
            system_prompt=system_prompt,
        )
        return trajectory
