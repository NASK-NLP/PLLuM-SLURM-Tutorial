import json
import inspect
import math
import re

import tools as calculator

TOOLS = {
    name: (function, tuple(inspect.signature(function).parameters), inspect.getdoc(function))
    for name, function in vars(calculator).items()
    if inspect.isfunction(function) and function.__module__ == calculator.__name__
}


def tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {arg: {"type": "number"} for arg in arguments},
                    "required": list(arguments),
                    "additionalProperties": False,
                },
            },
        }
        for name, (_, arguments, description) in TOOLS.items()
    ]


NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:[.,][0-9]*)?|[.,][0-9]+)(?:[eE][+-]?[0-9]+)?")
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("Number is outside floating-point range") from exc
    if not math.isfinite(number):
        raise ValueError("Expected a finite real number")
    return number


def execute_tool(call):
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Tool call must contain only name and arguments")
    name = call["name"]
    if not isinstance(name, str) or name not in TOOLS:
        raise ValueError(f"Unknown tool: {name!r}")
    function, expected_args, _ = TOOLS[name]
    arguments = call["arguments"]
    if not isinstance(arguments, dict) or set(arguments) != set(expected_args):
        raise ValueError(f"{name} requires arguments {', '.join(expected_args)}")
    try:
        values = [finite_number(arguments[arg]) for arg in expected_args]
        return finite_number(function(*values))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from exc


def parse_number(text):
    text = text.strip()
    if not NUMBER.fullmatch(text):
        raise ValueError("Expected only a final number")
    return finite_number(float(text.replace(",", ".")))


def parse_response(text):
    text = text.strip()
    match = TOOL_CALL.fullmatch(text)
    if match:
        call = json.loads(match.group(1))
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Tool call must contain only name and arguments")
        if not isinstance(call["name"], str) or not call["name"]:
            raise ValueError("Tool name must be a nonempty string")
        if not isinstance(call["arguments"], dict):
            raise ValueError("Tool arguments must be an object")
        for value in call["arguments"].values():
            finite_number(value)
        return call
    if "<tool_call>" in text or "</tool_call>" in text:
        raise ValueError("Expected exactly one tagged tool call with no additional text")
    return parse_number(text)
