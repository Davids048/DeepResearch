import json
from pathlib import Path
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt_openai_tools import TOOLS_OPENAI, TOOLS_GLM, TOOLS_GLM_PLAIN
from prompt import *
from prompt_builder import build_system_prompt, get_protocol_for_model, inject_prompt_section
import time
import asyncio

from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime

from logger import setup_logging
logger = setup_logging(name=__name__, level=15)


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.protocol = get_protocol_for_model(self.llm_local_path)

        logger.info(f"Initialized MultiTurnReactAgent with protocol: {self.protocol}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    # ==========================================
    # PROTOCOL METHODS (Wrappers)
    # ==========================================

    def call_server(self, msgs, planning_port, max_tries=10):
        """
        Protocol method: Call vLLM server with appropriate parameters.
        Dispatches to model-specific implementation based on detected protocol.
        """
        if self.protocol in ["minimaxm2"]:
            return self._call_server_openai_tools(msgs, planning_port, max_tries)
        elif self.protocol == "glm46":
            return self._call_server_plain(msgs, planning_port, max_tries)
        elif self.protocol == "default":
            return self._call_server_default(msgs, planning_port, max_tries)
        else:
            raise NotImplementedError(f"call_server not implemented for protocol: {self.protocol}")

    def parse_and_extract_tools(self, response, round):
        """
        Protocol method: Parse response and extract content + tool calls.
        Returns: (content_str, tool_calls_list)

        tool_calls_list format: [{"name": str, "arguments": dict}, ...]
        """
        if self.protocol in ["minimaxm2"]:
            return self._parse_and_extract_tools_openai_tools(response, round)
        elif self.protocol == "glm46":
            return self._parse_and_extract_tools_plain(response, round)
        elif self.protocol == "default":
            return self._parse_and_extract_tools_default(response, round)
        else:
            raise NotImplementedError(f"parse_and_extract_tools not implemented for protocol: {self.protocol}")

    def get_system_prompt(self):
        """Protocol method: Get appropriate system prompt for the model"""
        return build_system_prompt(model_name=self.llm_local_path)

    def count_tokens(self, messages):
        """Protocol method: Count tokens with model-specific template"""
        tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)

        if self.protocol in ["minimaxm2"]:
            # Include tools in token count for models using OpenAI tool format
            from prompt_openai_tools import TOOLS_OPENAI
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=TOOLS_OPENAI
            )
        elif self.protocol == "glm46":
            tpl = Path("template.jinja").read_text()
            tokenizer.chat_template = tpl
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=TOOLS_GLM
            )
        else:
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        return token_count

    # ==========================================
    # DEFAULT PROTOCOL IMPLEMENTATION
    # ==========================================

    def _call_server_default(self, msgs, planning_port, max_tries=10):
        """Default protocol: Standard vLLM chat completion"""
        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                logger.debug(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line below.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content

                if content and content.strip():
                    logger.debug("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    logger.warning(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.warning("Error: All retry attempts have been exhausted. The call has failed.")

        return f"vllm server error!!!"

    def _parse_and_extract_tools_default(self, content, round):
        """
        Default protocol: Parse JSON from <tool_call> tags.
        Returns: (content_str, tool_calls_list)
        """
        # Clean up <tool_response> if present
        if '<tool_response>' in content:
            pos = content.find('<tool_response>')
            content = content[:pos]

        # Extract tool calls
        tool_calls = []
        if '<tool_call>' in content and '</tool_call>' in content:
            tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
            try:
                # Handle Python special case
                if "python" in tool_call_str.lower():
                    try:
                        code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                        logger.debug(f"Round {round}: Detected Python code ({len(code_raw)} chars)")
                        tool_calls.append({
                            "name": "PythonInterpreter",
                            "arguments": {"code": code_raw}
                        })
                    except Exception as e:
                        logger.error(f"Round {round}: Python code extraction error: {e}")
                        tool_calls.append({
                            "name": "error",
                            "result": "[Python Interpreter Error]: Formatting error."
                        })
                else:
                    # Parse JSON tool call
                    tool_call = json5.loads(tool_call_str)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    logger.debug(f"Round {round}: Calling tool '{tool_name}' with args: {tool_args}")
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": tool_args
                    })
            except Exception as e:
                logger.error(f"Round {round}: Tool call parsing error - {str(e)[:100]}")
                tool_calls.append({
                    "name": "error",
                    "result": 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                })

        return content.strip(), tool_calls

    # ==========================================
    # OPENAI TOOLS PROTOCOL IMPLEMENTATION
    # Used by: MiniMax-M2, GLM-4.6
    # ==========================================
    def _call_server_plain(self, msgs, planning_port, max_tries=10):
        """OpenAI Tools protocol: Use vLLM/sglang tool calling API with automatic parsing"""
        # logger.critical("Using _call_server_plain")

        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1

        # Reformat msgs based on chat template. 
        tok = AutoTokenizer.from_pretrained("zai-org/GLM-4.6")
        tpl = Path("template.jinja").read_text()
        tok.chat_template = tpl
        prompt = tok.apply_chat_template(
            msgs,
            tools = TOOLS_GLM,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True,
        )
        logger.debug(f"agent prompt: \n{prompt}")
        

        for attempt in range(max_tries):
            try:
                # logger.debug(f"--- Attempting to call the service (OpenAI Tools), try {attempt + 1}/{max_tries} ---")
                response = client.completions.create(
                    model=self.model,
                    prompt = prompt,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                message = response.choices[0].text

                if message:
                    logger.debug("--- Service call successful, received a valid response ---")
                    logger.debug(f"appending <think> token for later parsing.")
                    return '<think>' + message
                else:
                    logger.warning(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.warning("Error: All retry attempts have been exhausted. The call has failed.")

        return None


    def _call_server_openai_tools(self, msgs, planning_port, max_tries=10):
        """OpenAI Tools protocol: Use vLLM/sglang tool calling API with automatic parsing"""
        from prompt_openai_tools import TOOLS_OPENAI

        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                logger.debug(f"--- Attempting to call the service (OpenAI Tools), try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    tools=TOOLS_OPENAI,
                    tool_choice="auto",
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                message = chat_response.choices[0].message

                if message:
                    logger.debug("--- Service call successful, received a valid response ---")
                    return message
                else:
                    logger.warning(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                logger.error(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)

                logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.warning("Error: All retry attempts have been exhausted. The call has failed.")

        return None

    def _parse_and_extract_tools_plain(self, text, round):
        from parse_tools_utils import parse_model_response
        parsed_response = parse_model_response(text, TOOLS_GLM_PLAIN)

        reasoning_content = parsed_response.get("reasoning_content", "")
        tool_calls = parsed_response.get("tool_calls", [])

        # example:
        # browser.search
        # {'query': '"compulsory" school reading 2017 African country novel'}
        return reasoning_content, tool_calls


    def _parse_and_extract_tools_openai_tools(self, message, round):
        """
        OpenAI Tools protocol: Extract content and tool calls from vLLM/sglang response.
        Used by: MiniMax-M2, GLM-4.6
        Returns: (content_str, tool_calls_list)
        """
        if message is None:
            logger.error(f"Round {round}: Received None from server")
            return "", []

        content = message.content or ""
        vllm_tool_calls = message.tool_calls

        # Convert vLLM tool calls to internal format
        tool_calls = []
        if vllm_tool_calls:
            logger.debug(f"Round {round}: vLLM parsed {len(vllm_tool_calls)} tool call(s)")
            for tc in vllm_tool_calls:
                function_name = tc.function.name
                try:
                    function_args = json.loads(tc.function.arguments)
                    logger.debug(f"Round {round}: Tool '{function_name}' with args: {function_args}")
                    tool_calls.append({
                        "name": function_name,
                        "arguments": function_args
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Round {round}: Failed to parse tool arguments for '{function_name}': {e}")
                    tool_calls.append({
                        "name": "error",
                        "result": f"Error parsing arguments for {function_name}"
                    })

        return content, tool_calls

    # ==========================================
    # UNIFIED EXECUTION LOOP (Protocol-agnostic)
    # ==========================================

    def _run(
        self,
        data: dict,
        model: str,
        system_prompt: str = None,
        **kwargs,
    ) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question

        # Use provided system_prompt or get protocol-specific prompt
        system_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        prompt_kwargs = {k:v for k,v in kwargs.items() if k in ['additional_instructions', 'knowledge_history']}
        user_prompt = inject_prompt_section(question, **prompt_kwargs)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # Log start of execution
        logger.info(f"=== Starting _run() for question: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info(f"Model: {model}, Protocol: {self.protocol}, Max LLM calls: {MAX_LLM_CALL_PER_RUN}")

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        answer_found = False
        prediction = ""
        while num_llm_calls_available > 0:
            # Check whether time is reached
            elapsed_time = time.time() - start_time
            if elapsed_time > 150 * 60:  # 150 minutes in seconds
                logger.warning(f"Time limit exceeded: {elapsed_time/60:.2f} minutes")
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            logger.debug(f"--- Round {round} starting (LLM calls remaining: {num_llm_calls_available}, elapsed: {elapsed_time/60:.1f}m) ---")

            # 1. Call server (protocol-specific)
            response = self.call_server(messages, planning_port)

            # 2. Parse and extract tools (protocol-specific)
            content, tool_calls = self.parse_and_extract_tools(response, round)
            assistant_msg = {
                "role": "assistant",
                "reasoning_content": content,
                "tool_calls": tool_calls,
            }
            # logger.debug(f"reasoning_content: {content}")
            # logger.debug(f"tool calls: {tool_calls}")

            # 3. Add assistant message
            messages.append(assistant_msg)

            # 4. Execute tool calls (protocol-agnostic)
            if tool_calls:
                tool_results = []
                for tc in tool_calls:
                    assert isinstance(tc, dict)
                    if tc["name"] == "error":
                        # Error during parsing
                        result = tc["result"]
                        logger.error(f"Round {round}: Tool parsing error: {result}")
                    elif tc["name"] == "finish":
                        logger.debug(f"Round {round}: Answer found.")
                        answer_found = True
                        result = "answer found"
                    else:
                        # Execute tool
                        try:
                            result = self.custom_call_tool(tc["name"], tc["arguments"])
                        except Exception as e:
                            result = f"Error calling tool {tc['name']}: {str(e)}"
                            logger.error(f"Round {round}: Tool execution error - {str(e)[:100]}")

                    tool_results.append(result)
                # Add to messages
                messages.append({
                    "role":"tool",
                    "content": [
                        {"output": out} for out in tool_results
                    ] 
                })

            # 5. Check for answer (protocol-agnostic)
            if answer_found:
                prediction = content
                break
            # if '<answer>' in content and '</answer>' in content:
            #     termination = 'answer'
            #     break

            # if num_llm_calls_available <= 0 and '<answer>' not in content:
            if num_llm_calls_available <= 0 and not answer_found:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'
                logger.warning(f"Round {round}: LLM call limit reached")

            # 6. Token limit checking (protocol-agnostic)
            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            logger.debug(f"round: {round}, token count: {token_count}")

            # Log token usage when approaching limit
            if token_count > max_tokens * 0.8:
                logger.warning(f"Round {round}: Token usage high - {token_count}/{max_tokens} ({token_count/max_tokens*100:.1f}%)")

            if token_count > max_tokens:
                logger.warning(f"Round {round}: Token limit exceeded - {token_count} > {max_tokens}")

                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"

                # Call server one more time to get final answer
                final_response = self.call_server(messages, planning_port)
                final_content, _ = self.parse_and_extract_tools(final_response, round)
                messages.append({"role": "assistant", "content": final_content})

                if '<answer>' in final_content and '</answer>' in final_content:
                    prediction = final_content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                    logger.warning(f"Final answer generated due to token limit: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")
                else:
                    prediction = final_content
                    termination = 'format error: generate an answer as token limit reached'
                    logger.error(f"Format error when generating answer due to token limit")

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        # Extract final prediction
        # if '<answer>' in messages[-1]['content']:
        #     prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
        #     termination = 'answer'
        if answer_found and prediction:
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        # Log final status
        total_elapsed = time.time() - start_time
        logger.info(f"=== _run() completed ===")
        logger.info(f"Termination: {termination}")
        logger.info(f"Total rounds: {round}")
        logger.info(f"Total elapsed time: {total_elapsed/60:.2f} minutes")
        logger.info(f"Prediction: {prediction[:150]}{'...' if len(prediction) > 150 else ''}")

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": round,
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            # tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
