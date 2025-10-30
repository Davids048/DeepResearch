from dataclasses import dataclass
import json
from typing import Any, Dict, Union, List
from evolve.llm import LLMClient
from evolve.playbook import Playbook
from evolve.reflector import ReflectorOutput
from evolve.delta import DeltaBatch
from evolve.curator_prompt import CURATOR_SYSTEM_PROMPT, CURATOR_TEMPLATE, CURATOR_TOOLS

from logger import setup_logging
logger = setup_logging(name=__name__, level=5)

@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """Transforms reflections into delta updates."""

    def __init__(
        self,
        llm: LLMClient,
        curator_system_prompt: str = CURATOR_SYSTEM_PROMPT,
        curator_template: str = CURATOR_TEMPLATE,
    ) -> None:
        self.llm = llm
        self.curator_system_prompt = curator_system_prompt
        self.curator_template = curator_template

    def curate(
        self,
        question_context: str,
        playbook: Playbook,
        reflector_output: ReflectorOutput,
    ) -> CuratorOutput:

        prompt = self.curator_template.format(
            question_context=question_context,
            current_playbook=playbook.as_prompt() or "(empty playbook)",
            current_reflections=reflector_output.raw,
        )

        logger.debug(f">>>>>>>>>>> curator received prompt:{prompt}.")
        response = self.llm.completion(
            messages=[
                {"role": "system", "content": self.curator_system_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=CURATOR_TOOLS,
        )
        logger.debug(f">>>>>>>>>> curator response:{response}.")
        try:
            curation_data = response.choices[0].message.tool_calls[0].function.arguments
            data = json.loads(curation_data)
        except Exception as e:
            logger.error(f"Unexpected error parsing curator response: {e}")
            logger.error(f"Raw response content: {response}")
            # Create a fallback data object for unexpected errors
            data = {
                "reasoning": "Failed.",
                "operations": [],
                "error": "Curator encountered unexpected error",
            }
            logger.warning("Using fallback curator output due to unexpected error")

        logger.debug(f"curator data:\n{data}.")

        delta = DeltaBatch.from_json(data)

        return CuratorOutput(delta=delta, raw=data)
