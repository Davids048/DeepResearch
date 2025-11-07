from evolve.reflector_prompt import (
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTION_TOOLS_KFLOW_PLAIN,
    REFLECTOR_TEMPLATE_KFLOW,
)


REFLECTION_SYSTEM_PROMPTS = {
    "base": REFLECTOR_SYSTEM_PROMPT,
    "v2": """You are an expert reviewer analyzing another assistant's reasoning trajectory.
Your goal is to extract insights to improve future task performance.

## Core role
- Interpret the trajectory as a *process*, not just the final output.
- Focus on reasoning quality, decision points, and evidence use.
- Think aloud systematically before outputting the reflection.

## Constraints
- Do not use tools.
- Base all analysis strictly on the given trajectory.
- Do not attempt to solve the underlying task.

## Output
- After completing your analysis, call `output_json_reflections` **once** with the structured result.
- Before return, double check that you have called `output_json_reflections` to output a structural tool_call.
"""
}


REFLECTION_USER_TEMPLATES = {
    "base": REFLECTOR_TEMPLATE_KFLOW,
    "v2": """
## Task
You will be given the assistant's full reasoning trajectory.
Your goal is to analyze it following the steps below:

0. Go over the assistant's trajectory round by round, summarize what happened in each round of interaction.
1. Identify major **decision points**, **hidden assumptions**, and **search strategies**.
2. Define **three evaluation rubrics** that capture key reasoning dimensions 
   (e.g., exploration depth, goal clarity, evidence reliability).
3. For each rubric, analyze performance **round by round**.
4. Summarize your observations to prepare for the structured reflection output.

## Data
### Assistant Trajectory
================================
{messages}
================================

### Final Assistant Prediction
================================
{prediction}
================================
"""
}


REFLECTION_TOOLS = {
    "base": REFLECTION_TOOLS_KFLOW_PLAIN,
    "v2":[{
        "name": "output_json_reflection",
        "description": "Output a structured reflection based on three self-defined aspects (rubrics).",
        "parameters": {
            "type": "object",
            "properties": {
                "aspects": {
                    "type": "array",
                    "description": "List of 3 reasoning dimensions (rubrics) defined by the model.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short title for the aspect, e.g. 'Exploration Depth'.",
                            },
                            "definition": {
                                "type": "string",
                                "description": "A concise explanation of what this aspect measures.",
                            }
                        },
                        "required": ["name", "definition"]
                    }
                },
                "aspect_reports": {
                    "type": "array",
                    "description": "Analytical commentary for each aspect, aligned by index with the 'aspects' array.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect_name": {"type": "string"},
                            "round_analysis": {
                                "type": "array",
                                "description": "One short note per major round/step in the trajectory describing how this aspect evolved.",
                                "items": {"type": "string"}
                            },
                            "summary_diagnosis": {
                                "type": "string",
                                "description": "1–3 sentence summary of strengths, weaknesses, or trends for this aspect."
                            }
                        },
                        "required": ["aspect_name", "summary_diagnosis"]
                    }
                },
                "overall_summary": {
                    "type": "string",
                    "description": "Brief overall reflection on the agent's process, synthesizing the three aspects."
                }
            },
            "required": ["aspects", "aspect_reports", "overall_summary"]
        }
    }]
}
