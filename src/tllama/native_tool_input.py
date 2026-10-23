"""Validate public tool declarations and match caller-provided history locally.

No schema execution or tool dispatch occurs here. Historical function names are
independent of the declarations available for the next inference turn.
"""

import json

from tllama.native_tools import argument_object
from tllama.worker import WorkerProtocolError


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(label + " must contain valid UTF-8") from exc
    return value


def _finite(value):
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("Tools data must contain finite JSON values") from exc


def tool_options(tools, choice=None, parallel=None):
    """Return native declarations/choice, without modifying caller data."""
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be an array")
    declarations, names = [], set()
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("tools supports only function declarations")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("Tool function must be an object")
        name = _name(function.get("name"), "Tool function name")
        if name in names:
            raise ValueError("Tool function names must be unique")
        names.add(name)
        for field, kind in (("description", str), ("parameters", dict), ("strict", bool)):
            if field in function and not isinstance(function[field], kind):
                raise ValueError(f"Tool function {field} must be {kind.__name__}")
        _finite(tool)
        declarations.append({"type": "function", **tool})
    if parallel is not None and type(parallel) is not bool:
        raise ValueError("parallel_tool_calls must be a boolean")
    if isinstance(choice, dict):
        function = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("Forced tool_choice needs type=function and function.name")
        name = _name(function.get("name"), "tool_choice function name")
        selected = [tool for tool in declarations if tool['function']['name'] == name]
        if len(selected) != 1:
            raise ValueError("Forced tool_choice must identify exactly one declared tool")
        # This native parser accepts auto/none/required, not the OpenAI object.
        declarations, choice = selected, "required"
    elif choice is not None and (not isinstance(choice, str) or choice not in {"auto", "none", "required"}):
        raise ValueError("tool_choice must be auto, none, required or a named function")
    if choice == "required" and not declarations:
        raise ValueError("tool_choice=required needs at least one declared tool")
    return declarations if tools is not None else None, choice


def tool_history(messages, *, api):
    """Validate and normalize already-copied messages before model admission.

    OpenAI matches explicit call IDs; Ollama also permits the documented
    name/ordered result form. Missing Ollama result IDs are filled only when
    the matched assistant call already supplied that ID. Never invent IDs.
    """
    if api not in {"openai", "ollama"}:
        raise ValueError("Unknown tool history API")
    pending = []
    for message in messages:
        role = message['role']
        calls = message.get('tool_calls') or []
        if calls and role != 'assistant':
            raise ValueError("Only assistant messages may contain tool_calls")
        if role != 'tool' and message.get('tool_call_id') is not None:
            raise ValueError("tool_call_id belongs only to a tool result")
        if role != 'tool' and pending:
            raise ValueError("Tool results must complete the preceding assistant calls")
        if calls:
            ids = set()
            for call in calls:
                if call.get('type', 'function') != 'function' or not isinstance(call.get('function'), dict):
                    raise ValueError("History supports only function tool calls")
                function = call['function']
                name = _name(function.get('name'), "Historical function name")
                identity = call.get('id')
                if identity is not None:
                    identity = _name(identity, "Historical tool call ID")
                    if identity in ids:
                        raise ValueError("Historical call IDs must be unique within an assistant turn")
                    ids.add(identity)
                elif api == 'openai':
                    raise ValueError("OpenAI historical tool calls require an ID")
                else:
                    # Optional Ollama IDs may be null in raw JSON; the native
                    # parser expects a missing field rather than a JSON null.
                    call.pop('id', None)
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    try:
                        argument_object(arguments)
                    except WorkerProtocolError as exc:
                        raise ValueError("Historical tool arguments must be complete object JSON") from exc
                elif isinstance(arguments, dict) and api == 'ollama':
                    _finite(arguments)
                else:
                    raise ValueError("Historical tool arguments must be an object (Ollama) or object JSON string")
                pending.append((identity, name))
        if role == 'tool':
            identity, name = message.get('tool_call_id'), message.get('name')
            if identity is not None:
                _name(identity, "Tool result ID")
            elif api == 'openai':
                raise ValueError("OpenAI tool results require tool_call_id")
            if name is not None:
                _name(name, "Tool result name")
            matches = [i for i, (call_id, call_name) in enumerate(pending)
                       if (identity is None or identity == call_id) and (name is None or name == call_name)]
            if not matches:
                raise ValueError("Tool result does not match a pending assistant call")
            # Ollama name-only results for repeated functions follow call order.
            call_id, call_name = pending.pop(matches[0])
            if api == 'ollama':
                message['name'] = call_name
                if call_id is not None:
                    message['tool_call_id'] = call_id
                else:
                    message.pop('tool_call_id', None)
    if pending:
        raise ValueError("Tool history is missing results for assistant calls")
    return messages
