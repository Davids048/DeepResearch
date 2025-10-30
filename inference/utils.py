from typing import List, Dict

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

        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        formatted_lines.append(f"Role: {role}")
        formatted_lines.append(f"Message: {content}")
        formatted_lines.append("")

    return "\n".join(formatted_lines)


def get_model_generation_config(model_name: str) -> Dict:
    """Get model-specific generation configuration.

    Args:
        model_name: Name or identifier of the model

    Returns:
        Dictionary with generation config parameters for the model
    """
    # Normalize model name for matching
    model_lower = model_name.lower()

    # GLM-4.6 (and variants)
    if "glm" in model_lower:
        return {
            "max_tokens": 16000,
            "temperature": 0.1,
            "top_p": 0.95,
        }

    # Default config for unknown models
    return {
        "max_tokens": 16000,
        "temperature": 1.0,
        "top_p": 0.95,
    }


