CURATOR_SYSTEM_PROMPT = """
You are a master curator of knowledge. 
Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

Context: 
- The playbook you created will be used to help improving future attempts or answering similar questions
- You need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

Instructions: 
- Review the existing playbook and the reflection from the previous attempt 
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook 
- Avoid redundancy: if similar advice already exists, only add new content that is a perfect complement to the existing playbook 
- Do NOT regenerate the entire playbook: only provide the operations needed 
- Focus on quality over quantity: a focused, well-organized playbook is better than an exhaustive one 
- For any operation if no new content to add, return an empty list for the operations field 
- Be concise and specific: each addition should be actionable 

Available Operations: 
1. ADD: Create new bullet points with fresh IDs 
    - section: the section to add the new bullet to 
    - content: the new content of the bullet. 
    Note: no need to include the bullet_id in the content like ‘[ctx-00263] helpful=1 harmful=0 ::’, the bullet_id will be added by the system.

Example 1:
Task Context: “Find money sent to roommates since Jan 1 this year” 
Current Playbook: [Basic API usage guidelines]  
Reflections: “The agent failed because it tried to identify roommates by parsing Venmo transaction descriptions instead of using the Phone app’s contact relationships. This led to incorrect identification and wrong results.”  Response:
{{
    "reasoning": "The reflection shows a critical error where the agent used unreliable heuristics (transaction descriptions) instead of the authoritative source (Phone app contacts) to identify relationships. This is a fundamental principle that should be captured in the playbook to prevent similar failures in identity resolution tasks.",
    "operations": [
        {{ 
            "type": "ADD",
                "section": "strategies_and_hard_rules",
                "content": "Always resolve identities from the correct source app\n- When you need to identify relationships (roommates, contacts, etc.), always use the Phone app's contact, and never try other heuristics from transaction descriptions, name patterns, or other indirect sources. These heuristics are unreliable and will cause incorrect results.",
        }}
    ],
}}

# Output format:
- After your analysis, you MUST use the 'output_json_curation' tool to produce a json object of the required operations.
- Before return, double check that the only tool you called is 'output_json_curation'
"""

CURATOR_TEMPLATE="""
Task Context (the actual task instruction):
{question_context}

Current Playbook:
{current_playbook}

Current Reflections (principles and strategies that helped to achieve current task):
{current_reflections}
"""

CURATOR_TOOLS = [{
    "type": "function",
    "function": {
        "name": "output_json_curation",
        "description": "Output a json object of the curation operations to apply to the playbook.",
        "parameters": {
            'type': 'object',
            'properties': {
                'reasoning': {
                    'type': 'string',
                    'description': "Step-by-step reasoning for the curation decisions"
                },
                'operations': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'type': {
                                'type': 'string',
                                'description': "Type of operation: ADD, UPDATE, TAG, or REMOVE",
                                'enum': ['ADD', 'UPDATE', 'TAG', 'REMOVE']
                            },
                            'section': {
                                'type': 'string',
                                'description': "Section of the playbook to modify"
                            },
                            'content': {
                                'type': 'string',
                                'description': "Content to add or update (optional for some operations)"
                            },
                            'bullet_id': {
                                'type': 'string',
                                'description': "ID of the bullet point to modify (required for UPDATE, TAG, REMOVE)"
                            },
                            'metadata': {
                                'type': 'object',
                                'description': "Metadata with helpful/harmful scores",
                                'additionalProperties': {'type': 'integer'}
                            }
                        },
                        'required': ['type', 'section'],
                        'additionalProperties': False,
                    },
                    'description': "List of delta operations to apply to the playbook",
                },
            },
            'required': ['reasoning', 'operations'],
            'additionalProperties': False,
        }
    }
},]
