import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from inference.evolve.reflector_prompt import REFLECTION_KFLOW_TOOLS, REFLECTION_TOOLS_KFLOW_PLAIN, REFLECTOR_SYSTEM_PROMPT
from inference.evolve.utils import get_glm_openai_client, get_glm_tokenizer
from inference.parse_tools_utils import parse_model_response
from logger import setup_logging

logger = setup_logging(name=__name__, level=10)


def generate_reflections(
    trajectory:dict,
    reflector_user_template,
    reflector_system_prompt,
    reflector_tools_plain,
    reflector_tools,
):
    messages = trajectory["messages"]
    prediction = trajectory["prediction"] 
    from evolve.reflector_prompt import REFLECTOR_TEMPLATE_KFLOW
    reflector_user_prompt = REFLECTOR_TEMPLATE_KFLOW.format(
        messages = messages,
        prediction = prediction,
    )
    tokenizer = get_glm_tokenizer()
    prompt = tokenizer.apply_chat_template(
       [
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": reflector_user_prompt},
        ],
        tools = REFLECTION_KFLOW_TOOLS,
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    client = get_glm_openai_client()
    def _generate_single_reflection():
        response = client.completions.create(
            model="zai-org/GLM-4.6",
            prompt = prompt,
            # Use following params for more variety.
            temperature=1.2,
            top_p = 0.85,
            seed = random.randint(1,1000),
            max_tokens=16000,
            # **DEFAULT_COMPLETION_CONFIG,
        )
        response = response.choices[0].text
        response = "<think>" + response
        logger.debug(f">>>>>>>>>> reflector response:{response}.")

        parsed_response = None
        try:
           # GLM is using pure text handling.
           parsed_response = parse_model_response(response, REFLECTION_TOOLS_KFLOW_PLAIN)
           reasoning_content = parsed_response.get("reasoning_content", "")
           tool_calls = parsed_response.get("tool_calls", [])
           data = tool_calls[0]["arguments"]
           data["reasoning_content"] = reasoning_content
        except Exception as e:
            logger.error(f"Unexpected error parsing reflector response, using fallback output. Error: {e}. Raw response:{parsed_response}")
            # Create a fallback data object for unexpected errors
            data = {
                "error": "Reflector encountered unexpected error",
            }
        return data 
    reflections = []
    num_reflections = int(os.getenv("MAX_REFLECTIONS", 16))           
    num_reflection_workers = int(os.getenv("MAX_REFLECTION_WORKERS", 16))           
    with ThreadPoolExecutor(max_workers = num_reflection_workers) as executor:
        futures = [executor.submit(_generate_single_reflection) for _ in range(num_reflections)]
        for future in as_completed(futures):
            try:
                reflections.append(future.result())
            except Exception as e:
                logger.error(f"Reflection future failed: {e}.")
                reflections.append({"error": "Reflection failed"})
    # Aggregate the results - pickout the ones where reflector judge the trace as wrong. 
    error_reflections = []
    for reflection in reflections:
        verdict = reflection.get("correctness_judgement", "")
        if verdict and verdict != "correct": # treating incorrect and incomplete as wrong.
            error_reflections.append(reflection)

