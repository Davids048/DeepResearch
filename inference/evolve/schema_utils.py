"""Utilities for generating JSON schemas from dataclasses for LLM structured outputs."""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Type, get_origin, get_args, get_type_hints


def get_json_type(field_type: Type, _nested_dataclass_schemas: Dict[Type, Dict] = None) -> Dict[str, Any]:
    """Convert Python type annotation to JSON schema type.

    Args:
        field_type: The Python type annotation
        _nested_dataclass_schemas: Internal cache for nested dataclass schemas

    Returns:
        Dict containing JSON schema type definition
    """
    if _nested_dataclass_schemas is None:
        _nested_dataclass_schemas = {}

    origin = get_origin(field_type)

    # Handle List types
    if origin is list or (isinstance(field_type, type) and issubclass(field_type, list)):
        args = get_args(field_type)
        if args:
            arg_type = args[0]
            # Check if it's a dataclass
            if is_dataclass(arg_type):
                # Generate schema for nested dataclass
                nested_schema = _create_dataclass_schema(arg_type, _nested_dataclass_schemas)
                return {"type": "array", "items": nested_schema}
            # Check if it's List[Dict[str, str]] or similar
            elif get_origin(arg_type) is dict:
                return {"type": "array", "items": {"type": "object"}}
            else:
                return {"type": "array", "items": get_json_type(arg_type, _nested_dataclass_schemas)}
        return {"type": "array", "items": {"type": "string"}}

    # Handle Dict types
    elif origin is dict or (isinstance(field_type, type) and issubclass(field_type, dict)):
        args = get_args(field_type)
        if args and len(args) == 2:
            key_type, val_type = args
            # For Dict[str, int] we can be more specific
            if val_type == int:
                return {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            elif val_type == str:
                return {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
        return {"type": "object"}

    # Handle Optional types
    elif origin is type(None):
        return {"type": "null"}

    # Handle basic types
    elif field_type == str or field_type == "str":
        return {"type": "string"}
    elif field_type == int or field_type == "int":
        return {"type": "integer"}
    elif field_type == float or field_type == "float":
        return {"type": "number"}
    elif field_type == bool or field_type == "bool":
        return {"type": "boolean"}

    # Handle dataclass types
    elif is_dataclass(field_type):
        return _create_dataclass_schema(field_type, _nested_dataclass_schemas)

    # Default to string
    else:
        # Fallback for string representation check
        type_str = str(field_type)
        if "List" in type_str:
            return {"type": "array", "items": {"type": "object"}}
        elif "Dict" in type_str:
            return {"type": "object"}
        return {"type": "string"}


def _create_dataclass_schema(dataclass_type: Type, cache: Dict[Type, Dict] = None) -> Dict[str, Any]:
    """Create a JSON schema object for a dataclass.

    Args:
        dataclass_type: The dataclass type to create schema for
        cache: Cache to avoid infinite recursion

    Returns:
        JSON schema object
    """
    if cache is None:
        cache = {}

    # Check cache to avoid infinite recursion
    if dataclass_type in cache:
        return cache[dataclass_type]

    # Create placeholder to handle circular references
    cache[dataclass_type] = {"type": "object"}

    properties = {}
    required = []

    # Use get_type_hints to resolve string annotations (from __future__ import annotations)
    try:
        type_hints = get_type_hints(dataclass_type)
    except Exception:
        # Fallback if get_type_hints fails
        type_hints = {}

    for f in fields(dataclass_type):
        # Get resolved type from type_hints, fallback to f.type
        field_type = type_hints.get(f.name, f.type)

        # Get type information
        type_info = get_json_type(field_type, cache)

        # Get description from metadata
        description = f.metadata.get("description", "")

        # Combine type info and description
        properties[f.name] = {
            **type_info,
            "description": description
        }

        # Add to required if no default value
        if f.default is f.default_factory is None:  # No default value
            required.append(f.name)

    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False
    }

    if required:
        schema["required"] = required

    cache[dataclass_type] = schema
    return schema


def create_response_format(
    output_class: Type,
    schema_name: str,
    required_fields: List[str] = None,
    exclude_fields: List[str] = None,
) -> Dict[str, Any]:
    """Create a response_format dict from a dataclass with field metadata.

    Args:
        output_class: Dataclass with fields that have metadata containing descriptions
        schema_name: Name for the JSON schema
        required_fields: List of required field names (defaults to all fields except excluded)
        exclude_fields: List of field names to exclude from schema (e.g., 'raw')

    Returns:
        Dict formatted for OpenAI-style response_format parameter

    Example:
        @dataclass
        class MyOutput:
            field1: str = field(metadata={"description": "First field"})
            field2: int = field(metadata={"description": "Second field"})
            raw: Dict = field(default_factory=dict)

        response_format = create_response_format(
            MyOutput,
            schema_name="my_schema",
            exclude_fields=["raw"]
        )
    """
    if not is_dataclass(output_class):
        raise ValueError(f"{output_class} must be a dataclass")

    exclude_fields = exclude_fields or []

    # Use get_type_hints to resolve string annotations (from __future__ import annotations)
    try:
        type_hints = get_type_hints(output_class)
    except Exception:
        # Fallback if get_type_hints fails
        type_hints = {}

    # Build properties
    properties = {}
    all_field_names = []

    for f in fields(output_class):
        if f.name in exclude_fields:
            continue

        all_field_names.append(f.name)

        # Get resolved type from type_hints, fallback to f.type
        field_type = type_hints.get(f.name, f.type)

        # Get type information
        type_info = get_json_type(field_type)

        # Get description from metadata
        description = f.metadata.get("description", "")

        # Combine type info and description
        properties[f.name] = {
            **type_info,
            "description": description
        }

    # Determine required fields
    if required_fields is None:
        required_fields = all_field_names

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False
            }
        }
    }
