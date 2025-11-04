from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Union, List

from transformers import AutoTokenizer
from evolve.llm import DEFAULT_COMPLETION_CONFIG, LLMClient
from evolve.playbook import Playbook
from evolve.reflector import ReflectorOutput
from evolve.delta import DeltaBatch
from evolve.curator_prompt import CURATOR_SYSTEM_PROMPT, CURATOR_TEMPLATE, CURATOR_TOOLS, CURATOR_TOOLS_PLAIN

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

        # Remove reflection reasoning content for less cluttering.
        current_reflections = reflector_output.raw 
        if "reasoning_content" in current_reflections.keys():
            current_reflections.pop("reasoning_content", None)

        prompt = self.curator_template.format(
            question_context=question_context,
            current_playbook=playbook.as_prompt() or "(empty playbook)",
            current_reflections=current_reflections,
        )

        logger.debug(f">>>>>>>>>>> curator received prompt:{prompt}.")
        if "glm" not in self.llm.model_name.lower():
            raise NotImplementedError()

        # Apply prompt 
        tok = AutoTokenizer.from_pretrained("zai-org/GLM-4.6")
        tpl = Path("template.jinja").read_text()
        tok.chat_template = tpl
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools = CURATOR_TOOLS,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
        # Plain generation
        response = self.llm.client.completions.create(
            model=self.llm.model_name,
            prompt = prompt,
            **DEFAULT_COMPLETION_CONFIG,
        )
        response = response.choices[0].text
        response = "<think>" + response

        logger.debug(f">>>>>>>>>> curator response:{response}.")
        try:
            # GLM is using pure text handling.
            from parse_tools_utils import parse_model_response
            parsed_response = parse_model_response(response, CURATOR_TOOLS_PLAIN)
            reasoning_content = parsed_response.get("reasoning_content", "")
            tool_calls = parsed_response.get("tool_calls", [])
            data = tool_calls[0]["arguments"]
            data["reasoning_content"] = reasoning_content
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
