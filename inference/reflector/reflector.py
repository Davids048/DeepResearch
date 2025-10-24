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

        reflection = self.reflect_single(
            question=question,
            prediction=prediction,
            messages=messages,
        )
        reflection_summary = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "termination": termination,
        }
        reflection_summary.update(reflection.raw)
        return reflection_summary




    def reflect_single(
        self,
        question,
        prediction:str,
        messages:List[dict],
    ) -> ReflectorOutput:
        """Create a single reflection on one agent task trajectory.
        Args:
            question: The original question posed to the agent.
            prediction: The final prediction made by the agent.
            messages: The list of messages in the agent's trajectory.
        Returns:
            ReflectorOutput: The reflection result. 
        """
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

        data = json.loads(response)
        return ReflectorOutput(
            reasoning=data.get("reasoning", ""),
            error_identification=data.get("error_identification", ""),
            root_cause_analysis=data.get("root_cause_analysis", ""),
            correct_approach=data.get("correct_approach", ""),
            key_insight=data.get("key_insight", ""),
            correctness_judgement=data.get("correctness_judgement", ""),
            raw=data,
        )

            



