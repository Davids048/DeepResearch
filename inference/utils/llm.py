from typing import List
from litellm import completion

class LLMClient:
    """Client to interact with Large Language Models (LLMs)."""
    def __init__(
        self, 
        model_name: str, 
        base_url:str, 
    ):
        self.model_name = model_name
        self.base_url = base_url

    def completion(self, messages: List, response_format:dict={}) -> str:
        """Generate a response from the LLM based on the given prompt."""
        raw_response = completion(
            model=f"hosted_vllm/{self.model_name}",
            messages=messages,
            base_url=self.base_url,
            max_tokens=16000,
            temperature=0.7,
            response_format=response_format,
        )
        response = raw_response.choices[0].message["content"]

        return response
