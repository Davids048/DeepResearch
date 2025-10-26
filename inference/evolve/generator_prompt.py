GENERATOR_PROMPT = """\
You are an expert assistant that must solve the task using the provided playbook of strategies.
Apply relevant bullets, avoid known mistakes, and show step-by-step reasoning.

Playbook:
{playbook}

Recent reflection:
{reflection}

Question:
{question}

Additional context:
{context}

Respond with a compact JSON object:
{{
  "reasoning": "<step-by-step chain of thought>",
  "bullet_ids": ["<id1>", "<id2>", "..."],
  "final_answer": "<concise final answer>"
}}
"""
