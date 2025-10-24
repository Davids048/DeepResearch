from dataclasses import dataclass, fields
import json
from typing import Dict, List, Any
from utils.llm import LLMClient
from reflector.reflector_prompt import REFLECTOR_TEMPLATE
from utils.logger import setup_logging

logger = setup_logging(level=5)

@dataclass
class ReflectorOutput: 
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    correctness_judgement: str
    bullet_tags: List[Dict[str, str]]
    raw: Dict[str, Any]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "reflection_schema",
        "schema": {
            "type": "object",
            "properties": {
                f.name: {"type": "string"} for f in fields(ReflectorOutput) if f.name != "raw"
            },
            "required": ["reasoning"],
            "additionalProperties": False
        }
    }
}

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
        trajectory
    ):
        question = trajectory['question']
        prediction = trajectory['prediction']
        messages = trajectory['messages']
        ground_truth = trajectory['answer']
        termination = trajectory['termination']

        # make a single reflection 
        prompt = self.reflection_template.format(
            question=question,
            prediction=prediction,
            messages=messages,  
        )
        response = self.llm.completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
        )
        logger.debug(f"Reflector LLM response: {response}...") 
        reflection = json.loads(response)

        
        reflection_summary = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "termination": termination,
        }
        reflection_summary.update(reflection)
        return reflection_summary

            



