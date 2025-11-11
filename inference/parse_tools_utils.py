import re
import json
import uuid

def parse_arguments(json_value):
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, isinstance(parsed_value, dict)
    except:
        return json_value, False

def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool["name"]: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    if arg_key not in tool["parameters"]["properties"]:
        return None
    return tool["parameters"]["properties"][arg_key]["type"]

def parse_model_response_reasoning_first(response: str, defined_tools: list):
    text = response.strip()
    reasoning_content = None
    content = None
    tool_calls = []
    
    # reasoning_content
    if text.startswith('<think>'):
        if '</think>' in text:
            reasoning_content, text = text.rsplit('</think>', 1)
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
            text = text.strip()
        else:
            reasoning_content = text.removeprefix('<think>').strip()
            text = ""
    
    # content      
    if '<tool_call>' in text:
        index = text.find('<tool_call>')
        content = text[:index].strip()
        text = text[index:].strip()
    else:
        content = text.strip()
        text = ""
    
    # tool_calls
    tool_call_strs = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    for call in tool_call_strs:
        func_name_match = re.match(r'([^\n<]+)', call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', call, re.DOTALL)
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = get_argument_type(func_name, arg_key, defined_tools)
                if arg_type != 'string':
                    arg_value, is_good_json = parse_arguments(arg_value)
                arguments[arg_key] = arg_value
                
            tool_calls.append({
                'tool_call_id': "tool-call-" + str(uuid.uuid4()),
                'name': func_name,
                'arguments': arguments
            })
    
    message = {'role': 'assistant'}
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    if content:
        message['content'] = content
    if tool_calls:
        message['tool_calls'] = tool_calls
    
    return message

def parse_model_response(response: str, defined_tools: list):
    """
    Alternative parsing that treats everything before the first <tool_call> as reasoning content.
    """
    text = response.strip()
    reasoning_content = None
    tool_calls = []

    # First, find all tool call blocks
    tool_call_strs = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)

    # Find the first <tool_call> tag
    if '<tool_call>' in text:
        index = text.find('<tool_call>')
        # Everything before the first <tool_call> is reasoning content
        reasoning_content = text[:index].strip()
        # Remove <think> tags if present
        if reasoning_content.startswith('<think>'):
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
        if reasoning_content.endswith('</think>'):
            reasoning_content = reasoning_content.removesuffix('</think>').strip()
    else:
        # No tool calls found, everything is reasoning content
        reasoning_content = text.strip()
        if reasoning_content.startswith('<think>'):
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
        if reasoning_content.endswith('</think>'):
            reasoning_content = reasoning_content.removesuffix('</think>').strip()

    # Parse tool calls
    for call in tool_call_strs:
        func_name_match = re.match(r'([^\n<]+)', call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>', call, re.DOTALL)
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = get_argument_type(func_name, arg_key, defined_tools)
                if arg_type != 'string':
                    arg_value, is_good_json = parse_arguments(arg_value)
                arguments[arg_key] = arg_value

            tool_calls.append({
                'tool_call_id': "tool-call-" + str(uuid.uuid4()),
                'name': func_name,
                'arguments': arguments
            })

    message = {'role': 'assistant'}
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    if tool_calls:
        message['tool_calls'] = tool_calls

    return message

def parse_model_response_json(response: str):
    """
    Parse model response where JSON is wrapped in ``` blocks.
    Everything before the json block is reasoning content.
    Use the last generated json block.
    The parsed json should be in the first tool call's arguments.
    """
    text = response.strip()
    reasoning_content = None
    tool_calls = []

    # Find all JSON code blocks (``` or ```json)
    json_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)

    if json_blocks:
        # Use the last JSON block
        last_json_block = json_blocks[-1].strip()

        # Find where the last JSON block starts in the text
        last_block_pattern = r'```(?:json)?\s*\n' + re.escape(last_json_block) + r'\n```'
        match = None
        for m in re.finditer(last_block_pattern, text, re.DOTALL):
            match = m

        if match:
            # Everything before the last JSON block is reasoning content
            reasoning_content = text[:match.start()].strip()
            # Remove <think> tags if present
            if reasoning_content.startswith('<think>'):
                reasoning_content = reasoning_content.removeprefix('<think>').strip()
            if reasoning_content.endswith('</think>'):
                reasoning_content = reasoning_content.removesuffix('</think>').strip()

        # Try to parse the JSON
        try:
            parsed_json = json.loads(last_json_block)
            # Create a tool call with the parsed JSON as arguments
            tool_calls.append({
                'tool_call_id': "tool-call-" + str(uuid.uuid4()),
                'name': 'json_response',  # Default name for JSON responses
                'arguments': parsed_json if isinstance(parsed_json, dict) else {'data': parsed_json}
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, treat everything as reasoning content
            reasoning_content = text
    else:
        # No JSON blocks found, everything is reasoning content
        reasoning_content = text
        if reasoning_content.startswith('<think>'):
            reasoning_content = reasoning_content.removeprefix('<think>').strip()
        if reasoning_content.endswith('</think>'):
            reasoning_content = reasoning_content.removesuffix('</think>').strip()

    message = {'role': 'assistant'}
    if reasoning_content:
        message['reasoning_content'] = reasoning_content
    if tool_calls:
        message['tool_calls'] = tool_calls

    return message 
