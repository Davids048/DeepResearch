from dataclasses import dataclass, field
import json
from typing import Dict, List, Any, Sequence
from evolve.llm import LLMClient
from evolve.reflector_prompt import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_TEMPLATE, REFLECTION_TOOLS
from evolve.schema_utils import create_response_format
from evolve.playbook import BulletTag, Playbook
from evolve.utils import format_messages
from logger import setup_logging

logger = setup_logging(name=__name__, level=5)

@dataclass
class ReflectorOutput:
    error_identification: str = field(
        metadata={"description": "What specifically went wrong in the reasoning? If reasoning is correct, state 'None'."},
    )
    root_cause_analysis: str = field(
        metadata={"description": "Why did this error occur? What concept was misunderstood?  If reasoning is correct, state 'None'."},
    )
    correct_approach: str = field(
        metadata={"description": "what should have been done (If reasoning is correct, what the agent do right)"},
    )
    key_insight: str = field(
        metadata={"description": "Reusable takeaways from this reflection"},
    )
    correctness_judgement: str = field(
        metadata={"description": "Judgement on the correctness of the prediction. Options: correct|incorrect|incomplete"},
    )
    bullet_tags: List[BulletTag] = field(
        metadata={"description": "List of {'id': '<bullet-id>', 'tag': 'helpful|harmful|neutral'}"},
    )
    raw: Dict[str, Any] = field(default_factory=dict)

# Generate response format from the dataclass
response_format = create_response_format(
    ReflectorOutput,
    schema_name="reflection_schema",
    exclude_fields=["raw"]
)

class Reflector:
    """Reflect on the task agent's trajectory."""
    def __init__(
        self,
        llm: LLMClient,
        reflection_template: str = REFLECTOR_TEMPLATE,
    ):
        self.llm = llm
        self.reflection_template = reflection_template

    def reflect(
        self,
        trajectory:Dict,
        playbook: Playbook,
    ):
        question = trajectory['question']
        prediction = trajectory['prediction']
        messages = trajectory['messages']
        answer = trajectory['answer']
        termination = trajectory['termination']
        consulted_playbook_section = "" #TODO: FILL THIS PART with generator's response

        # Format messages for better readability
        formatted_messages = format_messages(messages)

        # make a single reflection
        prompt = self.reflection_template.format(
            messages=formatted_messages,
            prediction=prediction,
            current_playbook=playbook.as_prompt() or "(empty playbook)",
            playbook_excerpt=consulted_playbook_section,
        )
            prediction=prediction,
        response = self.llm.completion(
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=REFLECTION_TOOLS,
        )
        reasoning, rest = self.llm.parse_response(response)
        try:
            data = json.loads(rest)
        except Exception as e:
            logger.error(f"Unexpected error parsing reflector response: {e}")
            logger.error(f"Raw response content: {rest}")
            # Create a fallback data object for unexpected errors
            data = {
                "error": "Reflector encountered unexpected error",
            }
            logger.warning("Using fallback reflector output due to unexpected error")

        logger.debug(f"reflector reasoning: {reasoning}.")
        logger.debug(f"reflector json:\n{data}.")

        # Create bullet tags
        bullet_tags: List[BulletTag] = []
        tags_payload = data.get("bullet_tags", [])
        if isinstance(tags_payload, Sequence):
            for item in tags_payload:
                if isinstance(item, dict) and "id" in item and "tag" in item:
                    bullet_tags.append(
                        BulletTag(
                            id=str(item["id"]), tag=str(item["tag"]).lower()
                        )
                    )

        reflector_output = ReflectorOutput(
            error_identification=str(data.get("error_identification", "")),
            root_cause_analysis=str(data.get("root_cause_analysis", "")),
            correct_approach=str(data.get("correct_approach", "")),
            key_insight=str(data.get("key_insight", "")),
            correctness_judgement=str(data.get("correctness_judgement", "")),
            bullet_tags=bullet_tags,
            raw=data,
        )
        if not (bullet_tags or reflector_output.key_insight):
            logger.warning("No bullet tags or key_insights created...")


        return reflector_output

            



