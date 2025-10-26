import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from evolve.llm import LLMClient
from evolve.generator_prompt import GENERATOR_PROMPT
from react_agent import MultiTurnReactAgent
from evolve.playbook import Playbook

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

    def generate(
        self,
        task: dict,
        playbook: Playbook = None,
        reflection: Optional[str] = None,
        debug: bool = False,
    ):
        if debug:
            logger.debug("Bypass actual generation. Use a mock response ")
            # 
            trajectory_path = "/home/hal-jundas/agent/DeepResearch/inference/output/Qwen3-235B-A22B-Thinking-2507/browsecomp/20251024-184158/iter1.evolved.jsonl"
            with open(trajectory_path, "r") as f:
                raw_data = f.readline()
                data = json.loads(raw_data)
                trajectory = {
                    "question": data.get("question"),
                    "answer": data.get("answer"),
                    "messages": data.get("messages"),
                    "prediction": data.get("prediction"),
                    "termination": data.get("termination")
                }
            return trajectory


        base_prompt = """""" # TODO: Follow ace's suggestion to add base prompt

        trajectory = self.task_agent._run(
            data=task,
            model=self.model_name,
        )
        return trajectory


