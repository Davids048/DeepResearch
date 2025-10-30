REFLECTOR_SYSTEM_PROMPT = """
You are an expert AI agent specialized in analyzing and reflecting on the performance of other AI agents in multi-step agentic tasks. 
Your task is to identify possible errors, inefficiencies, and areas for improvement in the agent's reasoning and action process.

### Key instructions:
- You may not use any tools given to the agent. 
- You must ground your analysis in the agent's existing trace, without using any other external sources.
- You must use the tool "output_json_reflection" at the end of your analysis. 

# Hard rules:
- You should not try to solve the problem by yourself. Instead, you should ground your analysis in the model's reasoning and action trajectory.
- You should not try to find the correct answer yourself, or use any tools to facilitate the analysis.

# Output format:
- After your analysis, you MUST use the 'output_json_reflection' tool to produce a json object of your reflection
- Before return, double check that the only tool you called is 'output_json_reflection'
"""


REFLECTOR_TEMPLATE = """
Help me analyze the following trajectory step by step. Then answer whether the agent successfully completed the task.

### Agent's Message Trajectory:
===============================================================================
{messages}
===============================================================================
Agent's Message End

### Final agent Prediction:
{prediction}

### Current playbook:
{current_playbook}

### Playbook excerpts consulted:
{playbook_excerpt}
"""


REFLECTION_TOOLS = [{
    "type": "function",
    "function": {
        "name": "output_json_reflection",
        "description": "Output a json object of the reflection on the task agent's trajectory.",
        "parameters": {
            'type': 'object',
            'properties': {
                'reasoning': {
                    'type': 'string',
                    'description': "Your reasoning here."
                },
                'error_identification': {
                    'type': 'string',
                    'description': "What specifically went wrong in the reasoning? If reasoning is correct, state 'None'.",
                },
                'root_cause_analysis': {
                    'type': 'string',
                    'description': "Why did this error occur? What concept was misunderstood?  If reasoning is correct, state 'None'.",
                },
                'correct_approach': {
                    'type': 'string',
                    'description': 'what should have been done (If reasoning is correct, what the agent do right)',
                },
                'key_insight': {
                    'type': 'string',
                    'description': 'Reusable takeaways from this reflection',
                },
                'correctness_judgement': {
                    'type': 'string',
                    'description': 'Judgement on the correctness of the prediction. Options: correct|incorrect|incomplete',
                },
                'bullet_tags': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {'id': {'type': 'string', 'description': ''}, 'tag': {'type': 'string', 'description': ''}},
                        'additionalProperties': False,
                    },
                    'description': "List of {'id': '<bullet-id>', 'tag': 'helpful|harmful|neutral'}",
                },
            },
            'required': ['error_identification', 'root_cause_analysis', 'correct_approach', 'key_insight', 'correctness_judgement', 'bullet_tags'],
            'additionalProperties': False,
        }
    }
},]

