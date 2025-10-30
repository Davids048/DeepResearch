"""
Composable System Prompt Builder

This module provides utilities to build system prompts by composing modular sections:
- Task description (context)
- Tools and usage instructions (model-specific)
- Playbook section (optional, for evolver)
- Reflection section (optional, for evolver)
"""

from typing import Optional
from prompt import TASK_DESCRIPTION, TOOLS_SECTION_DEFAULT


def detect_protocol(model_name: str) -> str:
    """Detect which protocol/model family to use based on model name."""
    model_lower = model_name.lower()

    if "minimax" in model_lower or "m2" in model_lower:
        return "minimaxm2"
    elif "glm" in model_lower:
        return "glm46"
    # Add future models here
    # elif "qwen3" in model_lower:
    #     return "qwen3"

    return "default"


def get_task_description() -> str:
    """Get the base task description (same for all models)."""
    return TASK_DESCRIPTION


def get_tools_section(protocol: str) -> str:
    """
    Get the tools section for a protocol.

    For minimaxm2: tools are passed via API, so return empty string
    For glm46: tools are passed via API, so return empty string
    For default: return the XML tool definitions that go in the prompt
    """
    if protocol == "minimaxm2":
        # Tools passed via vLLM API, not in prompt
        return ""

    elif protocol == "glm46":
        # Tools passed via vLLM API, not in prompt
        return ""

    elif protocol == "default":
        # Tools embedded in prompt as XML
        return TOOLS_SECTION_DEFAULT

    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def format_playbook_section(playbook) -> str:
    """Format the playbook into a prompt section."""
    playbook_text = playbook.as_prompt()
    if not playbook_text:
        return ""

    section = "\n\n# Strategic Playbook\n"
    section += "The following playbook contains proven strategies and common pitfalls. Apply relevant guidance during your research:\n"
    section += f"\n{playbook_text}"

    return section


def format_reflection_section(reflection: str) -> str:
    """Format the reflection into a prompt section."""
    if not reflection or not reflection.strip():
        return ""

    section = "\n\n# Recent Reflection\n"
    section += "Learn from the following reflection on your previous attempt:\n"
    section += f"\n{reflection}"

    return section


def build_system_prompt(
    model_name: str,
    playbook = None,
    reflection: Optional[str] = None,
) -> str:
    """
    Build a system prompt by composing modular sections.

    Args:
        model_name: Full model name/path (used to detect protocol)
        playbook: Optional Playbook instance to include strategic guidance
        reflection: Optional reflection text from previous iteration

    Returns:
        Complete system prompt string
    """
    protocol = detect_protocol(model_name)
    sections = []

    # 1. Task description
    sections.append(get_task_description())

    # 2. Tools section (protocol-specific)
    tools_section = get_tools_section(protocol)
    if tools_section:
        sections.append(tools_section)

    # 3. Playbook section (optional)
    if playbook:
        playbook_section = format_playbook_section(playbook)
        if playbook_section:
            sections.append(playbook_section)

    # 4. Reflection section (optional)
    if reflection:
        reflection_section = format_reflection_section(reflection)
        if reflection_section:
            sections.append(reflection_section)

    sections.append("\nUsing the information above, solve the following task:\n")

    # Compose all sections
    return "\n".join(sections)


def get_protocol_for_model(model_name: str) -> str:
    """Get the protocol identifier for a given model."""
    return detect_protocol(model_name)
