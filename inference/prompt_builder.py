"""
Composable System Prompt Builder

This module provides utilities to build system prompts by composing modular sections: - Task description (context) Tools and usage instructions (model-specific)
- Playbook section (optional, for evolver)
- Reflection section (optional, for evolver)
"""

from typing import List, Optional
from prompt import TASK_DESCRIPTION, TASK_DESCRIPTION_GLM, TOOLS_SECTION_DEFAULT


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


def get_task_description(protocol:str) -> str:
    if protocol == "glm46":
        return TASK_DESCRIPTION_GLM
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

def format_knowledge_section(knowledge_history:List[dict])->str:
    if not knowledge_history:
        return ""

    section = "\n\n# Previous review list:"
    for i, k in enumerate(knowledge_history):
        if isinstance(k, dict):
            # Extract the summarized proposed adjustments from the compression output
            summarized_adjustments = k.get("summarized_proposed_adjustments", "")
            if summarized_adjustments:
                section += (
                    f"\n## review {i}:\n"
                    f"{summarized_adjustments}\n"
                )
            # If no valid content, skip this entry (don't add anything)
        elif isinstance(k, str):
            section += (
                f"\n## review {i}:\n"
                f"{k}\n"
            )
    return section


def build_system_prompt(
    model_name: str,
    playbook = None,
    reflection: Optional[str] = None,
    knowledge_history = None,
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
    sections.append(get_task_description(protocol))

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

    if knowledge_history:
        knowledge_section = format_knowledge_section(knowledge_history)
        if knowledge_section:
            sections.append(knowledge_section)

    # Compose all sections
    return "\n".join(sections)

def inject_prompt_section(base_prompt, **kwargs):
    """Inject a base prompt with new information.
    This can be system prompt or user prompt.

    Args:
        base_prompt: The existing prompt to inject sections into
        **kwargs: Optional sections to inject 

    Returns:
        Modified prompt with injected sections appended
    """
    sections = [base_prompt]

    if 'additional_instructions' in kwargs and kwargs['additional_instructions']:
        sections.append(kwargs['additional_instructions'])

    # Inject knowledge_history section if provided
    if 'knowledge_history' in kwargs and kwargs['knowledge_history']:
        knowledge_section = format_knowledge_section(kwargs['knowledge_history'])
        if knowledge_section:
            sections.append(knowledge_section)

    return "\n".join(sections)


def get_protocol_for_model(model_name: str) -> str:
    """Get the protocol identifier for a given model."""
    return detect_protocol(model_name)

def tools_plain2openai(tools:List[dict]):
    return [
        {
            "type": "function",
            "function": tool,
        } for tool in tools
    ]
