from typing import List, Dict

def format_messages(messages: List[Dict]) -> str:
    """Format messages from trajectory into a structured, readable format for the reflector.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted string representation of the entire conversation history
    """
    formatted_lines = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        formatted_lines.append(f"Role: {role}")
        formatted_lines.append(f"Message: {content}")
        formatted_lines.append("")

    return "\n".join(formatted_lines)


