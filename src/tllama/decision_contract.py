"""Audited text Decision metadata and typed native request/response checks."""

import json
import math
from collections.abc import Mapping


DECISION_TYPES = frozenset({'laya', 'nimble', 'clef'})
# Text Decision reviewed independently on MI50/Vulkan0; see runtime-audit-070.md.
DECISION_RUNTIME = 'b11850-0e9433bfd3'
SUPPORTED_DECISION_RUNTIME_FINGERPRINTS = frozenset({DECISION_RUNTIME, 'b11552-3144c167ef'})
MAX_QUESTIONS = 32
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def has_decision_head(metadata: Mapping) -> bool:
    raw = metadata.get('metadata_raw') or {}
    return f"{raw.get('general.architecture')}.decision.type" in raw


def decision_type(metadata: Mapping) -> str | None:
    """Select a head from GGUF, never from a public model or file name."""
    raw = metadata.get('metadata_raw') or {}
    arch = raw.get('general.architecture')
    key = f'{arch}.decision.type'
    if key not in raw:
        return None
    kind = raw[key]
    if not isinstance(kind, str) or kind not in DECISION_TYPES:
        raise ValueError(f'Decision type {kind!r} is not integrated; audited types are Laya, Nimble and Clef')
    if arch != {'laya': 'modern-bert', 'nimble': 'qwen35', 'clef': 'clef'}[kind]:
        raise ValueError('Decision architecture differs from the audited head contract')
    template = raw.get('tokenizer.chat_template.systemone')
    if not isinstance(template, str) or not template:
        raise ValueError('Decision model requires its embedded systemone template')
    for name, value in raw.items():
        if name.startswith(f'{arch}.decision.temperature.') and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError('Decision temperature must be finite and positive')
    if kind == 'laya':
        blocks, head = raw.get(f'{arch}.block_count'), raw.get(f'{arch}.decision.block_count')
        budget = raw.get(f'{arch}.decision.max_head_tokens')
        if (type(blocks) is not int or type(head) is not int or not 0 < head < blocks
                or type(budget) is not int or budget <= 0):
            raise ValueError('Laya requires valid Decision head blocks and token budget')
        vocab = (metadata.get('embedding') or {}).get('vocab_size')
        for suffix in ('mask_token_id', 'seperator_token_id'):
            token = raw.get('tokenizer.ggml.' + suffix)
            if type(vocab) is not int or type(token) is not int or not 0 <= token < vocab:
                raise ValueError('Laya requires valid MASK/SEP vocabulary IDs')
    return kind


def decision_capable(metadata: Mapping) -> bool:
    """Offline text capability; native startup still verifies the actual head."""
    if metadata.get('embedding_configured') is True or metadata.get('has_projector') is True:
        return False
    try:
        return decision_type(metadata) is not None
    except ValueError:
        return False


def prepare_decision_request(payload: Mapping) -> dict:
    """Snapshot literal JSON before any await; no template or generation options."""
    if not isinstance(payload, Mapping):
        raise ValueError('Decision request must be an object')
    body = dict(payload)
    if set(body) - {'model', 'state', 'questions'}:
        raise ValueError('Text Decision accepts only model, state and questions')
    if 'state' not in body or body['state'] is None:
        raise ValueError('Decision state must be provided and non-null')
    questions = body.get('questions')
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f'Decision questions must contain 1..{MAX_QUESTIONS} entries')
    for q in questions.values():
        if not isinstance(q, dict) or set(q) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Decision question accepts only type, instructions and criteria')
        if 'instructions' not in q or q['instructions'] is None:
            raise ValueError('Decision instructions must be provided and non-null')
        kind, criteria = q.get('type'), q.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError('Decision choice criteria must contain 1..255 options')
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError('Decision score criteria must contain 2..10 levels')
        elif kind == 'noul':
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError('Decision noul criteria must be an object or null')
        else:
            raise ValueError('Decision question type must be choice, score or noul')

    # Bound depth/nodes as well as bytes before serializing arbitrary JSON.
    pending, nodes = [(body, 0)], 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 100000:
            raise ValueError('Decision JSON exceeds the depth/node limit')
        if type(value) is dict:
            if len(value) > 100000:
                raise ValueError('Decision JSON exceeds the node limit')
            if any(type(key) is not str for key in value):
                raise ValueError('Decision JSON object keys must be strings')
            pending.extend((v, depth + 1) for v in value.values())
        elif type(value) is list:
            if len(value) > 100000:
                raise ValueError('Decision JSON exceeds the node limit')
            pending.extend((v, depth + 1) for v in value)
        elif type(value) not in (str, int, float, bool, type(None)):
            raise ValueError('Decision values must be JSON values')
        elif type(value) is float and not math.isfinite(value):
            raise ValueError('Decision JSON numbers must be finite')
    # Pinned parse_state() extracts media from these two chat-shaped layouts,
    # even without top-level images/files. Refuse that native interpretation
    # for text workers; similarly named data elsewhere stays literal JSON.
    state = body['state']
    messages = state.get('messages') if isinstance(state, dict) and 'messages' in state else state
    if isinstance(messages, list):
        for message in messages:
            content = message.get('content') if isinstance(message, dict) else None
            if isinstance(content, list) and any(isinstance(part, dict) and (
                    part.get('type') == 'image_url' and 'image_url' in part
                    or part.get('type') == 'input_audio' and 'input_audio' in part) for part in content):
                raise ValueError('Media parts in Decision chat state are not supported by the text contract')
    try:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Decision request is not valid UTF-8 JSON') from exc
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError('Decision request exceeds the 2 MiB JSON limit')
    return json.loads(encoded)


def validate_decision_response(payload: dict, questions: dict, kind: str, alias: str) -> dict:
    """Verify native accounting and formulas without changing any returned value."""
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    def number(value, low, high):
        need(type(value) in (int, float) and low <= value <= high and math.isfinite(value),
             'Decision numeric value is not finite or in range')
        return value

    need(payload.get('model') == alias, 'Decision response model differs from worker alias')
    answers, usage = payload.get('answers'), payload.get('usage')
    need(isinstance(answers, dict) and list(answers) == list(questions), 'Decision answer IDs/order differ')
    need(isinstance(usage, dict) and type(usage.get('input_tokens')) is int and usage['input_tokens'] > 0
         and type(usage.get('output_tokens')) is int and usage['output_tokens'] == 0, 'Invalid Decision usage')
    for key, question in questions.items():
        answer = answers[key]
        need(isinstance(answer, dict) and answer.get('type') == question['type'], 'Decision answer type differs')
        if question['type'] == 'noul':
            number(answer.get('noul'), 0, 1)
            continue
        keys = list(question['criteria']) if question['type'] == 'choice' else list(map(str, range(len(question['criteria']))))
        if kind == 'clef' and question['type'] == 'choice':
            keys.sort()
        probabilities = answer.get('probabilities')
        need(isinstance(probabilities, dict) and list(probabilities) == keys, 'Decision probability keys/order differ')
        probs = [number(probabilities[k], 0, 1) for k in keys]
        need(abs(math.fsum(probs) - 1) < 1e-4, 'Decision probabilities do not sum to one')
        n = len(probs)
        if question['type'] == 'choice':
            choice = answer.get('choice')
            need(isinstance(choice, str) and choice in keys and probabilities[choice] >= max(probs) - 1e-7,
                 'Decision choice is not an argmax')
            confidence = 1 if n == 1 else max(0, (max(probs) - 1/n) / (1 - 1/n))
        else:
            # JSON booleans must not compare equal to integer legend values.
            need(json.dumps(answer.get('legend'), sort_keys=True, allow_nan=False)
                 == json.dumps(dict(zip(keys, question['criteria'])), sort_keys=True, allow_nan=False),
                 'Decision score legend differs')
            expected = math.fsum(i * p for i, p in enumerate(probs))
            need(abs(number(answer.get('score'), 0, n - 1) - expected) < 1e-4, 'Decision score formula differs')
            mode = probs.index(max(probs))
            distance = math.fsum(p * abs(i - mode) for i, p in enumerate(probs))
            uniform = math.fsum(abs(i - (n - 1)/2)/n for i in range(n))
            confidence = max(0, 1 - distance / uniform)
        need(abs(number(answer.get('confidence'), 0, 1) - confidence) < 1e-4, 'Decision confidence formula differs')
    return payload
