from typing import List, Dict
import json
import re
from openai import OpenAI

from logger import setup_logging 
logger = setup_logging(name=__name__, level=5)

DEFAULT_COMPLETION_CONFIG = {
    "max_tokens": 16000,
    "temperature": 0.7,
    # "top_p": 1.0,
    # "frequency_penalty": 0.0,
    # "presence_penalty": 0.0,
}

class LLMClient:
    """Client to interact with Large Language Models (LLMs)."""
    def __init__(
        self, 
        model_name: str, 
        base_url:str, 
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
        )


    def completion(
        self,
        messages: List,
        completion_config=DEFAULT_COMPLETION_CONFIG,
        **kwargs,
    ) -> str:
        """Generate a response from the LLM based on the given prompt."""

        raw_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **completion_config,
            **kwargs,
        )
        return raw_response

    def parse_reasoning(self, message_text):

        if not message_text:
            return None, message_text

        reasoning_match = re.match(r"<(?:think|thinking)>(.*?)</(?:think|thinking)>(.*)", message_text, re.DOTALL)

        if reasoning_match:
            return reasoning_match.group(1), reasoning_match.group(2)

        # 2) Fallback: closing tag only
        print("Trying fallback...")
        if "</think>" in message_text or "</thinking>" in message_text:
            logger.debug(f"falling back to parse only closing think tag.")
            parts = re.split(r"</(?:think|thinking)>", message_text, maxsplit=1)
            reasoning = parts[0].strip()
            content = parts[1] if len(parts) > 1 else ""
            return reasoning if reasoning else None, content

        return None, message_text

    def parse_response(self, response):
        message = response.choices[0].message

        # Check if reasoning_content is already separated
        if hasattr(message, 'reasoning_content') and message.reasoning_content is not None:
            return message.reasoning_content, message.content

        # Otherwise, parse the content to extract reasoning
        content = message.content if hasattr(message, 'content') else ""
        reasoning, remaining_content = self.parse_reasoning(content)
        return reasoning, remaining_content 

