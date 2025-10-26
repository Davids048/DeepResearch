from dataclasses import dataclass
import json
from typing import Any, Dict, Union, List
from evolve.llm import LLMClient
from evolve.playbook import Playbook
from evolve.reflector import ReflectorOutput
from evolve.delta import DeltaBatch
from evolve.curator_prompt import CURATOR_PROMPT
from evolve.schema_utils import create_response_format

from logger import setup_logging
logger = setup_logging(name=__name__, level=5)

@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]

# Generate response format from DeltaBatch dataclass
response_format = create_response_format(
    DeltaBatch,
    schema_name="curator_schema",
    # required_fields=["reasoning"],
    exclude_fields=[]
)


class Curator:
    """Transforms reflections into delta updates."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template

    def curate(
        self,
        question_context: str,
        playbook: Playbook,
        reflector_output: ReflectorOutput,
    ) -> CuratorOutput:

        prompt = self.prompt_template.format(
            response_format=response_format,
            question_context=question_context,
            current_playbook=playbook.as_prompt() or "(empty playbook)",
            current_reflections=reflector_output.raw,
        )

        response = self.llm.completion(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            response_format=response_format,
        )
        logger.debug(f"Curator response: {response}")
        reasoning, rest = self.llm.parse_response(response)
        data = json.loads(rest)
        delta = DeltaBatch.from_json(data)

        return CuratorOutput(delta=delta, raw=data)
