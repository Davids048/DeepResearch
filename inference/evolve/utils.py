from pathlib import Path
from typing import List, Dict
from openai import OpenAI

from transformers import AutoTokenizer

def format_messages(messages: List[Dict]) -> str:
    """Format messages from trajectory into a structured, readable format for the reflector.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted string representation of the entire conversation history
    """
    formatted_lines = []
    for i, msg in enumerate(messages):
        if i == 0:
            # skip system message
            continue
        formatted_lines.append(f"{msg}")

    return "\n".join(formatted_lines)


def get_glm_tokenizer():
    tok = AutoTokenizer.from_pretrained("zai-org/GLM-4.6")
    tpl = Path("template.jinja").read_text()
    tok.chat_template = tpl
    return tok


def get_glm_openai_client():
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:6000/v1",
    )

