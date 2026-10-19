"""Small TCP subprocess implementing the pinned protocol for lifecycle tests."""

import asyncio
import base64
import hashlib
import json
import math
import os
import signal
import sys
import struct
import zlib
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODE = os.environ.get("STUB_MODE", "normal")
ROOT = Path(os.environ["STUB_ROOT"])
KEY = Path(arg("--api-key-file")).read_text().strip()
LOCK = asyncio.Lock()
HEALTH_READS = 0
TIMINGS = {"cache_n": 19, "prompt_n": 6, "prompt_ms": 2.5,
           "predicted_n": 10, "predicted_ms": 200.1}
if os.environ.get('STUB_PUBLIC_MTP') and arg('--spec-type') == 'draft-mtp':
    TIMINGS.update(draft_n=12, draft_n_accepted=6)
USAGE = {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35,
         "prompt_tokens_details": {"cached_tokens": 19}}


def image_reply(body):
    parts = [part for message in body.get('messages', []) if isinstance(message.get('content'), list)
             for part in message['content'] if part.get('type') == 'image_url']
    if not parts:
        return None
    # Decode the test/smoke PNG's actual RGB panels independently of its prompt.
    data = base64.b64decode(parts[-1]['image_url']['url'].split(',', 1)[1])
    width = struct.unpack('>I', data[16:20])[0]
    compressed, offset = b'', 8
    while offset < len(data):
        size = struct.unpack('>I', data[offset:offset+4])[0]
        if data[offset+4:offset+8] == b'IDAT':
            compressed += data[offset+8:offset+8+size]
        offset += size + 12
    row = zlib.decompress(compressed)
    names = {b'\xff\0\0':'red', b'\0\0\xff':'blue'}
    return json.dumps({'left':names.get(row[1:4], 'unknown'),
                       'right':names.get(row[1+width//2*3:4+width//2*3], 'unknown')})


def mark(name, value):
    with (ROOT / name).open("a") as file:
        file.write(json.dumps(value) + "\n")


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode()


async def json_response(writer, value, status=200):
    body = encode(value)
    writer.write(f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    await writer.drain()


async def embeddings_request(reader, writer, body):
    """Synthetic embeddings only; no model or numerical-quality simulation."""
    mode = os.environ.get('STUB_EMBEDDINGS', 'normal')
    if (ROOT / 'embedding-behavior').exists():
        mode = (ROOT / 'embedding-behavior').read_text()
    texts = body.get('input')
    token_rows = isinstance(texts, list) and texts and all(isinstance(row, list) for row in texts)
    token_input = isinstance(texts, list) and texts and type(texts[0]) is int
    if not texts:
        await json_response(writer, {'error': {'message': 'empty input or context exceeded'}}, 400)
        return
    if body.get('encoding_format', 'float') not in {'float', 'base64'}:
        await json_response(writer, {'error': {'message': 'invalid encoding'}}, 500 if mode == 'error500' else 400)
        return
    token_count = len(texts) if token_input else None
    if os.environ.get('STUB_PUBLIC_EMBEDDINGS'):
        # The public diagnostic compares native text with final token arrays.
        # Use the same synthetic tokenizer for both independent input paths.
        if token_input:
            texts, token_rows = [texts], True
        elif not token_rows:
            items = [texts] if isinstance(texts, str) else texts
            texts = [[1, *(value + 10 for value in text.encode('utf-8')), 2] for text in items]
            token_rows = True
    lengths = list(map(len, texts)) if token_rows else [len(texts)] if token_input else []
    context = int(arg('--ctx-size', '256'))
    exclusive = os.environ.get('STUB_EMBEDDING_POOLING') == '3'
    if any(n > context or exclusive and n == context for n in lengths):
        await json_response(writer, {'error': {'message': 'context exceeded'}}, 400)
        return
    if token_rows:
        token_count = sum(map(len, texts))
        texts = [json.dumps(row) for row in texts]
    elif token_input:
        texts = ['token ids']
    elif isinstance(texts, str):
        texts = [texts]
    if mode in {'http400', 'http500'}:
        await json_response(writer, {'error': {'message': 'inference failed'}}, 400 if mode == 'http400' else 500)
        return
    if mode == 'hold' or len(texts) == 32 and mode != 'fast':
        mark('started', 'embeddings')
        await reader.read()
        mark('cancelled', 'embeddings')
        return
    data = []
    for index, text in enumerate(texts):
        vector = [float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:8]]
        if mode == 'drift' and len(texts) > 1:
            vector[0] += 100
        norm = math.sqrt(sum(value * value for value in vector))
        if body.get('embd_normalize') != -1:
            vector = [value / norm for value in vector]
        if mode == 'wrong-dimension' and index:
            vector = vector[:-1]
        if mode == 'bad-norm':
            vector = [value * 2 for value in vector]
        if mode == 'nan':
            vector[0] = float('nan')
        data.append({'object': 'embedding', 'index': 0 if mode == 'bad-index' else index, 'embedding': vector})
    tokens = token_count if token_count is not None else sum(len(text) + 1 for text in texts)
    payload = {'object': 'list', 'data': data, 'model': arg('--alias'),
               'usage': {'prompt_tokens': tokens, 'total_tokens': tokens}}
    if mode == 'no-usage':
        del payload['usage']
    if mode == 'bad-usage':
        payload['usage']['prompt_tokens'] += 1
    if mode == 'reverse':
        payload['data'].reverse()
    await json_response(writer, payload)


async def multimodal_embeddings_request(reader, writer, body):
    """Independent fixture decoder for the native multimodal diagnostic."""
    mode = os.environ['STUB_MULTIMODAL_EMBEDDINGS']
    if (ROOT / 'media-behavior').exists():
        mode = (ROOT / 'media-behavior').read_text().strip()
    value = body.get('input')
    if not value or body.get('encoding_format', 'float') not in {'float', 'base64'}:
        await json_response(writer, {'error': {'message': 'invalid request'}}, 400)
        return
    token_input = isinstance(value, list) and type(value[0]) is int
    entries = [value] if not isinstance(value, list) or token_input else value
    if (len(entries) == 8 and entries == [entries[0]] * 8 and mode != 'fast') or (ROOT / 'media-hold').exists():
        mark('started', 'multimodal-embeddings')
        await reader.read()
        mark('cancelled', 'multimodal-embeddings')
        return

    def pixels(encoded):
        data = base64.b64decode(encoded, validate=True)
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('invalid PNG')
        offset, compressed = 8, b''
        while offset < len(data):
            size = struct.unpack('>I', data[offset:offset + 4])[0]
            if data[offset + 4:offset + 8] == b'IDAT':
                compressed += data[offset + 8:offset + 8 + size]
            offset += size + 12
        return zlib.decompress(compressed)

    def canonical(entry):
        if isinstance(entry, str):
            return entry.encode()
        if isinstance(entry, list):
            if os.environ.get('STUB_PUBLIC_EMBEDDINGS'):
                return bytes(t - 10 for t in entry[1:-1])
            return ''.join(chr(t - 10) for t in entry[1:]).encode()
        chunks = []
        if 'content' in entry:
            for part in entry['content']:
                if part['type'] == 'text':
                    chunks.append(part['text'].encode())
                elif part['type'] == 'image_url':
                    image = pixels(part['image_url']['url'].split(',', 1)[1])
                    chunks.append(b'' if mode == 'ignore-images' else b'IMAGE:' + image)
                else:
                    raise ValueError('unsupported content')
        else:
            parts = entry['prompt_string'].split('<stub-media>')
            chunks.append(parts[0].encode())
            for part, data in zip(parts[1:], entry['multimodal_data']):
                image = pixels(data)
                chunks.extend([b'' if mode == 'ignore-images' else b'IMAGE:' + image, part.encode()])
        return b''.join(chunks)

    data = []
    try:
        for index, entry in enumerate(entries):
            digest = hashlib.sha256(canonical(entry)).digest()
            vector = [float(v + 1) for v in digest[:8]]
            if mode == 'drift' and len(entries) > 1:
                vector[0] += 100
            norm = math.sqrt(sum(v * v for v in vector))
            vector = [v / norm for v in vector]
            if mode == 'bad-norm':
                vector = [v * 2 for v in vector]
            if mode == 'wrong-dimension':
                vector.pop()
            if mode == 'nan':
                vector[0] = float('nan')
            if body.get('encoding_format') == 'base64':
                vector = base64.b64encode(struct.pack('<' + 'f' * len(vector), *vector)).decode()
            data.append({'object': 'embedding', 'index': 99 if mode == 'out-of-range' else 0 if mode == 'bad-index' else index,
                         'embedding': vector})
    except (ValueError, KeyError, zlib.error) as exc:
        invalid_png = str(exc) == 'invalid PNG'
        message = ('Failed to load image or audio file' if invalid_png else 'invalid media/content')
        if invalid_png and mode == 'decoder-error':
            message = 'unrelated decoder failure'
        await json_response(writer, {'error': {'message': message}}, 500 if invalid_png else 400)
        return
    def token_count(entry):
        if isinstance(entry, list):
            return len(entry)
        if isinstance(entry, dict):
            if 'content' in entry:
                if any(p['type'] == 'image_url' for p in entry['content']):
                    return 50
                entry = ''.join(p['text'] for p in entry['content'])
            else:
                if entry['multimodal_data']:
                    return 50
                entry = entry['prompt_string']
        return len(entry.encode()) + 2 if os.environ.get('STUB_PUBLIC_EMBEDDINGS') else len(entry) + 1
    count = sum(token_count(e) for e in entries)
    if (ROOT / 'media-token-count').exists():
        count = int((ROOT / 'media-token-count').read_text())
    payload = {'object': 'list', 'data': data, 'model': arg('--alias'),
               'usage': {'prompt_tokens': count, 'total_tokens': count}}
    if mode in {'bad-usage', 'over-limit', 'boolean-usage'}:
        payload['usage'] = {'prompt_tokens': True, 'total_tokens': True} if mode == 'boolean-usage' else {
            'prompt_tokens': 2000 if mode == 'over-limit' else count, 'total_tokens': 2000 if mode == 'over-limit' else count + 1}
    if mode == 'no-usage':
        del payload['usage']
    await json_response(writer, payload)


async def tools_request(reader, writer, body):
    """Synthetic Tools protocol only; does not simulate real model quality."""
    mode = os.environ['STUB_TOOLS']
    if mode in {'http400','http500'}:
        await json_response(writer,{'error':{'message':'template does not support this request'}},
                            400 if mode == 'http400' else 500)
        return
    messages = body['messages']
    outputs = [json.loads(m['content'])['marker'] for m in messages if m['role'] == 'tool']
    calls, content = [], ''
    if outputs:
        content = ' '.join(outputs)
    elif body.get('tool_choice', 'auto') == 'none':
        content = 'NO_TOOL'
    else:
        prompt = next(m['content'] for m in reversed(messages) if m['role'] == 'user')
        keys = json.loads(prompt.split(': ',1)[1])
        if isinstance(keys,str):
            keys = [keys]
        for index,key in enumerate(keys):
            calls.append({'id':f'call-{index}','type':'function','function':{
                'name':'lookup_record','arguments':json.dumps({'key':'wrong' if mode == 'semantic-bad' else key},ensure_ascii=False)}})
    finish = 'tool_calls' if calls else 'stop'
    if not body['stream']:
        await json_response(writer,{'choices':[{'index':0,'message':{'role':'assistant','content':content,'tool_calls':calls},
                                               'finish_reason':finish}],'usage':USAGE,'timings':TIMINGS})
        return
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
    await writer.drain()
    def frame(delta, reason=None):
        return event({'choices':[{'index':0,'delta':delta,'finish_reason':reason}]})
    if calls:
        for index,call in enumerate(calls):
            await stream_bytes(writer,frame({'tool_calls':[{'index':index,'id':call['id'],'type':'function',
                'function':{'name':'lookup' if mode == 'growing-name' else call['function']['name']}}]}))
            if mode == 'growing-name':
                await stream_bytes(writer,frame({'tool_calls':[{'index':index,'id':call['id'],
                    'function':{'name':call['function']['name']}}]}))
        # Interleave calls; split JSON strings and escapes independently of TCP.
        arguments = [call['function']['arguments'] for call in calls]
        for offset in range(0,max(map(len,arguments)),3):
            for index,text in enumerate(arguments):
                if offset < len(text):
                    part = 'not-json' if mode == 'malformed-json' else text[offset:offset+3]
                    await stream_bytes(writer,frame({'tool_calls':[{'index':index,'function':{'arguments':part}}]}))
            if 'cancel_' in messages[-1]['content']:
                mark('started','tools')
                await reader.read()
                mark('cancelled','tools')
                return
        if mode == 'truncated':
            return
    else:
        await stream_bytes(writer,frame({'content':content}))
    await stream_bytes(writer,frame({},finish))
    await stream_bytes(writer,event({'choices':[],'usage':USAGE,'timings':TIMINGS}))
    if mode != 'no-done':
        await stream_bytes(writer,b'data: [DONE]\r\n\r\n')


async def stream_bytes(writer, value):
    # Split UTF-8 code points, delimiters and JSON across writes.
    for offset in range(0, len(value), 3):
        writer.write(value[offset:offset + 3])
        await writer.drain()
        await asyncio.sleep(0)


async def mtp_request(reader, writer, body):
    """Synthetic MTP counters/transport only, never a model or speed benchmark."""
    mode = os.environ['STUB_MTP']
    enabled = arg('--spec-type') == 'draft-mtp'
    if mode == 'http500':
        await json_response(writer, {'error': 'synthetic MTP failure'}, 500)
        return
    prompt = body['messages'][-1]['content']
    content = ('READY' if 'READY' in prompt else '1, 2, 3' if 'sequence' in prompt else
               'def merge_sorted(a, b):\n    return a + b  # synthetic only')
    if mode == 'different-text' and enabled:
        content += ' changed'
    if mode == 'punctuated-control' and 'READY' in prompt:
        content = 'READY.'
    if (os.environ.get('STUB_MTP_NEGATIVE') == 'changed-recovery' and not enabled
            and len((ROOT / 'launch').read_text().splitlines()) == 3):
        content = 'CHANGED'
    if 'READY' in prompt:
        global MTP_CONTROL_READS
        MTP_CONTROL_READS = globals().get('MTP_CONTROL_READS', 0) + 1
        if mode == 'changed-control' and enabled and MTP_CONTROL_READS > 1:
            content = 'CHANGED'
    timings = {**TIMINGS, 'predicted_per_second': 90.0 if enabled else 45.0}
    if enabled and mode != 'no-drafting':
        timings.update(draft_n=12, draft_n_accepted=6 if mode != 'bad-counts' else 13)
    finish = None if mode == 'no-finish' else 'stop'
    state_mode = os.environ.get('STUB_MTP_STATE')
    if state_mode:
        cached = body.get('cache_prompt') is True
        timings['cache_n'] = 19 if cached and state_mode != 'no-reuse' else 0
        if state_mode == 'unknown-cache':
            timings.pop('cache_n')
        if enabled and cached and state_mode == 'stale-state':
            content += ' STALE'
        if enabled and cached and state_mode == 'wrong-finish':
            finish = 'length'
        if enabled and cached and state_mode == 'wrong-count':
            timings['predicted_n'] += 1
        if state_mode == 'unknown-count':
            timings.pop('predicted_n')
    if not body['stream']:
        await json_response(writer, {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': finish}], 'timings': timings, 'usage': USAGE})
        return
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
    await writer.drain()
    for piece in content:
        await stream_bytes(writer, event({'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]}))
        if '10000' in prompt:
            mark('started', 'mtp')
            await reader.read()
            mark('cancelled', 'mtp')
            return
    await stream_bytes(writer, event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}))
    await stream_bytes(writer, event({'choices': [], 'timings': timings, 'usage': USAGE}))
    if mode != 'no-done':
        await stream_bytes(writer, b'data: [DONE]\r\n\r\n')


def event(value):
    data = json.dumps(value, ensure_ascii=False)
    split, quoted, escaped = -1, False, False
    for index, char in enumerate(data):
        if escaped:
            escaped = False
        elif quoted and char == '\\':
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == ',' and not quoted:
            split = index
            break
    if split != -1:
        return ("event: message\r\ndata: " + data[:split + 1] +
                "\r\ndata: " + data[split + 1:] + "\r\n\r\n").encode()
    return ("data: " + data + "\r\n\r\n").encode()


FIM_TOKENS = {"[PRE]": 1100001, "[SUF]": 1100002, "[MID]": 1100003,
              "[REP]": 1100004, "[SEP]": 1100005, "<bos>": 7,
              "<|fim_prefix|>": 1100011, "<|fim_suffix|>": 1100012, "<|fim_middle|>": 1100013,
              "<|file_separator|>": 70, "<fim_prefix>": 1100021,
              "<fim_suffix>": 1100022, "<fim_middle>": 1100023, "<file_sep>": 1100024,
              "[PREFIX]": 1100031, "[SUFFIX]": 1100032, "[MIDDLE]": 1100033,
              "<｜fim▁begin｜>": 1100041, "<｜fim▁hole｜>": 1100042, "<｜fim▁end｜>": 1100043,
              "<|file_sep|>": 1100051, "<|repo_name|>": 1100052, "<|fim_pad|>": 1100053}

# Small IDs permit real, compact GGUF headers in public-route tests. Reference
# diagnostic fixtures retain their independent historical token assignments.
if os.environ.get('STUB_PUBLIC_FIM'):
    FIM_TOKENS = {piece: index for index, piece in enumerate(FIM_TOKENS, 1)}


def fim_selected():
    mode = os.environ.get('STUB_FIM')
    pieces = (('<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>') if mode in {'codegemma', 'codegemma-split', 'codegemma-bad-piece', 'qwen'} else
              ('<fim_prefix>', '<fim_suffix>', '<fim_middle>') if mode == 'starcoder' else
              ('[PRE]', '[SUF]', '[MID]'))
    return dict(zip(('PRE', 'SUF', 'MID'), pieces))


def fim_tokenize(text, special=False):
    result = []
    if text.startswith('☺'):
        result.extend([8, 9])
        text = text[1:]
    while text:
        piece = next((p for p in FIM_TOKENS if special and text.startswith(p)), None)
        if piece:
            result.append(FIM_TOKENS[piece])
            text = text[len(piece):]
        else:
            result.append(ord(text[0]) + 1000)
            text = text[1:]
    return result


async def fim_request(reader, writer, path, body):
    mode = os.environ['STUB_FIM']
    if path == '/tokenize':
        tokens = (body['content'] if isinstance(body['content'], list) else
                  fim_tokenize(body['content'], body.get('parse_special', True)))
        if mode in {'codegemma-split', 'codegemma-bad-piece'} and body['content'] == '<|file_separator|>':
            tokens = fim_tokenize(' <|file_separator|>', special=False)
        if mode == 'codegemma' and body['content'] == '<|file_separator|>':
            tokens = [1032] + tokens
        tokens = ([FIM_TOKENS['<bos>']] if body.get('add_special') else []) + tokens
        if body.get('with_pieces'):
            reverse = {value: key for key, value in FIM_TOKENS.items()}
            tokens = [{'id': token, 'piece': reverse[token] if token in reverse else chr(token - 1000)} for token in tokens]
            if (mode == 'codegemma-bad-piece' and isinstance(body['content'], list)
                    and FIM_TOKENS['<|file_separator|>'] in body['content']):
                tokens[0]['piece'] = 'WRONG'
        await json_response(writer, {'tokens': tokens})
        return
    if path == '/infill' and mode in {'negative', 'missing-route', 'codestral'}:
        message = ('Infill is not supported by this model: prefix token is missing. suffix token is missing. middle token is missing.'
                   if mode != 'missing-route' else 'not found')
        await json_response(writer, {'error': {'message': message, 'code': 501 if mode != 'missing-route' else 404}},
                            501 if mode != 'missing-route' else 404)
        return
    async with LOCK:
        if path == '/infill':
            batch = int(arg('--batch-size'))
            pre, suf = fim_tokenize(body['input_prefix']), fim_tokenize(body['input_suffix'])
            selected = fim_selected()
            pre, suf = [FIM_TOKENS[selected['PRE']]] + pre[-3 * (batch // 4):], [FIM_TOKENS[selected['SUF']]] + suf[:batch // 4 - 2]
            tokens = [7] + (suf + pre if '--spm-infill' in sys.argv else pre + suf) + [FIM_TOKENS[selected['MID']]]
            if mode in {'repo', 'qwen'}:
                tokens = [1100004] + fim_tokenize('myproject\n') + [1100005] + fim_tokenize('filename\n') + tokens
        else:
            tokens = body['prompt'] if isinstance(body['prompt'], list) else fim_tokenize(body['prompt'])
        reverse = {value: key for key, value in FIM_TOKENS.items()}
        text = ''.join(reverse[t] if t in reverse else '' if t in (8, 9) else chr(t - 1000) for t in tokens)
        if '// Read the ' in text or '# Read the "' in text:
            content = 'amber' if 'Read the amber' in text or 'Read the "amber"' in text else 'violet'
        else:
            content = 'a * b' if '== 14' in text else 'a + b'
        if (ROOT / 'fim-content').exists():
            content = (ROOT / 'fim-content').read_text()
        if mode == 'semantic-bad':
            content = 'a - b'
        stopping_word = ''
        generated = fim_tokenize(content)
        if body.get('logit_bias'):
            forced = (int(next(iter(body['logit_bias']))) if isinstance(body['logit_bias'], dict)
                      else body['logit_bias'][0][0])
            piece = reverse[forced]
            generated = [forced]
            if piece in body.get('stop', []) and piece in body.get('preserved_tokens', []) and mode != 'ignored-stops':
                content, stopping_word = '', piece
            else:
                content = piece + 'unstopped'
        elif mode in {'codegemma', 'codegemma-split', 'codegemma-bad-piece', 'starcoder'}:
            piece = '<|file_separator|>' if mode in {'codegemma', 'codegemma-split', 'codegemma-bad-piece', 'qwen'} else '<file_sep>'
            generated.append(FIM_TOKENS[piece])
            if piece in body.get('stop', []) and piece in body.get('preserved_tokens', []):
                stopping_word = piece
            else:
                content += piece + 'unwanted continuation'
                generated.extend(fim_tokenize('unwanted continuation'))
        final = {'content': content, 'stop': True, 'stop_type': 'word' if stopping_word else 'eos',
                 'stopping_word': stopping_word, 'tokens': generated, 'tokens_evaluated': len(tokens),
                 'tokens_predicted': len(generated), 'timings': {'prompt_n': len(tokens), 'predicted_n': len(generated)}}
        if not body['stream']:
            await json_response(writer, final)
            return
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
        await writer.drain()
        await stream_bytes(writer, event({'content': content, 'stop': False}))
        if (body.get('ignore_eos') or body.get('stub') == 'hold') and not body.get('logit_bias'):
            mark('started', 'fim-stream')
            await reader.read()
            mark('cancelled', 'fim-stream')
        elif mode != 'bad-terminal':
            await stream_bytes(writer, event({**final, 'content': ''}))


async def handle(reader, writer):
    global HEALTH_READS
    try:
        request = await reader.readline()
        if not request:
            return
        path = request.decode().split()[1]
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body_bytes = await reader.readexactly(int(headers.get("content-length", "0")))
        body = json.loads(body_bytes) if body_bytes else {}
        mark("requests", {"path": path, "body": body, "header_names": sorted(headers)})
        if headers.get("authorization") != "Bearer " + KEY:
            await json_response(writer, {"error": "unauthorized"}, 401)
            return
        if path == "/health":
            HEALTH_READS += 1
            if MODE == "loading" and HEALTH_READS <= 3:
                await json_response(writer, {"error": "Loading model"}, 503)
            else:
                await json_response(writer, {"status": "bad" if MODE == "bad-health" else "ok"})
            return
        if path == "/props":
            await json_response(writer, {
                "build_info": (None if MODE == "missing-build" else "wrong" if MODE == "wrong-build"
                               else os.environ.get("STUB_BUILD", "b11552-ec2797836f")),
                "model_alias": "wrong" if MODE == "wrong-alias" else arg("--alias"),
                "total_slots": 4 if MODE == "wrong-slots" else 1,
                "model_path": arg("--model"),
                "default_generation_settings": {"n_ctx": 0 if MODE == "bad-context" else int(os.environ.get('STUB_PUBLIC_CTX',2048))},
                "modalities": {"vision": "--mmproj" in sys.argv and MODE != 'no-vision'},
                "media_marker": "<stub-media>",
                "bos_token": "<bos>", "eos_token": "<eos>",
                "chat_template": (Path(arg("--chat-template-file")).read_text() if "--chat-template-file" in sys.argv
                                  else "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"),
            })
            return
        if path == "/tllama/memory":
            behavior = (ROOT / "memory-behavior").read_text() if (ROOT / "memory-behavior").exists() else "normal"
            if behavior == "hang":
                await reader.read()
                mark("cancelled", "memory")
                return
            if behavior == "disconnect":
                return
            status = int((ROOT / "memory-status").read_text()) if (ROOT / "memory-status").exists() else 200
            file = ROOT / "memory-report"
            if not file.exists():
                await json_response(writer, {"error": "not found"}, 404)
                return
            data = file.read_bytes()
            writer.write(f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
            await writer.drain()
            return
        if os.environ.get('STUB_FIM') and path in {'/tokenize', '/completion', '/infill'}:
            await fim_request(reader, writer, path, body)
            return
        if os.environ.get('STUB_TOOLS') and path == '/v1/chat/completions':
            async with LOCK:
                await tools_request(reader,writer,body)
            return
        if os.environ.get('STUB_MTP') and path == '/v1/chat/completions':
            async with LOCK:
                await mtp_request(reader, writer, body)
            return
        if os.environ.get('STUB_MULTIMODAL_EMBEDDINGS') and path == '/v1/embeddings':
            await multimodal_embeddings_request(reader, writer, body)
            return
        if (os.environ.get('STUB_EMBEDDINGS') or '--embeddings' in sys.argv) and path == '/v1/embeddings':
            await embeddings_request(reader, writer, body)
            return
        if path == '/tokenize' and os.environ.get('STUB_PUBLIC_EMBEDDINGS'):
            tokens = [value + 10 for value in body['content'].encode('utf-8')]
            if body.get('add_special'):
                tokens = [1, *tokens, 2]
            await json_response(writer, {'tokens': tokens})
            return
        if path == "/tokenize":
            # Deterministic IDs expose exact render text in the request log.
            text = body["content"]
            tokens = [ord(char) + 10 for char in text]
            if text.startswith("<bos>"):
                tokens = [1] + [ord(char) + 10 for char in text[5:]]
            if body.get("add_special"):
                tokens.insert(0, 1)
            await json_response(writer, {"tokens": [True] if MODE == 'bad-tokenize' else tokens})
            return
        async with LOCK:
            # The public smoke's long stream must still be running when its
            # client observes /api/ps and closes the connection.
            held = body.get('ignore_eos') or body.get('stream') and body.get('max_tokens', body.get('n_predict')) == 512
            mode = body.get("stub", "hold" if held else "normal")
            if mode == "crash":
                os._exit(71)
            if mode == "error":
                await json_response(writer, {"error": {"code": 400, "message": "bad request"}}, 400)
                return
            chat = path == "/v1/chat/completions"
            if MODE == 'bad-image':
                await json_response(writer, {'error':{'code':400, 'message':'Image decoder rejected the payload'}}, 400)
                return
            vision_text = image_reply(body)
            if mode == "hold" and not body["stream"]:
                mark("started", "non-stream")
                await reader.read()
                mark("cancelled", "non-stream")
                return
            if not body["stream"] or mode == "json-stream":
                constrained = 'response_format' in body or 'json_schema' in body
                text = vision_text or ('{"answer":42}' if constrained else "Ahoj světe")
                if chat:
                    result = {"choices": [{"message": {"content": text, "reasoning_content": "myšlenka"},
                                           "finish_reason": "stop"}], "usage": USAGE, "timings": TIMINGS}
                else:
                    result = {"content": text, "stop": True, "stop_type": "eos", "timings": TIMINGS,
                              "tokens_evaluated": 25, "tokens_predicted": 10, "tokens_cached": 35}
                await json_response(writer, result)
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            await stream_bytes(writer, b": ping\r\n\r\n")
            chunk = {"choices": [{"delta": {"content": vision_text or "Ahoj světe"}, "finish_reason": None}]} if chat else {
                "content": "Ahoj světe", "stop": False}
            await stream_bytes(writer, event(chunk))
            if mode == "hold":
                mark("started", "stream")
                await reader.read()
                mark("cancelled", "stream")
                return
            if mode == 'controlled':
                mark('started', 'controlled')
                disconnected = asyncio.create_task(reader.read())
                try:
                    while not (ROOT / 'release').exists():
                        if disconnected.done():
                            mark('cancelled', 'controlled')
                            return
                        await asyncio.sleep(0.01)
                finally:
                    disconnected.cancel()
                    await asyncio.gather(disconnected, return_exceptions=True)
            if mode == "bad-json":
                await stream_bytes(writer, b"data: {broken}\n\n")
                return
            if mode == "error-stream":
                await stream_bytes(writer, event({"error": {"code": 500, "message": "decode failed"}}))
                return
            if mode == "truncated":
                return
            if chat:
                await stream_bytes(writer, event({"choices": [{"delta": {"reasoning_content": "myšlenka"},
                                                               "finish_reason": None}]}))
                await stream_bytes(writer, event({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
                await stream_bytes(writer, event({"choices": [], "usage": USAGE, "timings": TIMINGS}))
                if mode == "finish-without-done":
                    return
                await stream_bytes(writer, b"data: [DONE]\r\n\r\n")
            else:
                await stream_bytes(writer, event({"content": "", "stop": True, "stop_type": "eos", "timings": TIMINGS,
                                                  "tokens_evaluated": 25, "tokens_predicted": 10, "tokens_cached": 35}))
                if mode == "hold-after-stop":
                    mark("started", "raw-terminal")
                    await reader.read()
                    mark("cancelled", "raw-terminal")
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def main():
    mark("launch", {"argv": sys.argv[1:], "cwd": os.getcwd(),
                    "cwd_mode": oct(Path.cwd().stat().st_mode & 0o777),
                    "key_mode": oct(Path(arg("--api-key-file")).stat().st_mode & 0o777),
                    "env": {k: v for k, v in os.environ.items() if k.startswith(("LLAMA_", "AIP_"))
                            or k in ("CUDA_VISIBLE_DEVICES", "GGML_BACKEND_PATH", "MTMD_BACKEND_DEVICE")}})
    if MODE == "fatal":
        print("device kernel image is invalid", file=sys.stderr, flush=True)
        return 23
    negative = os.environ.get('STUB_MTP_NEGATIVE')
    if negative and arg('--spec-type') == 'draft-mtp' and negative != 'unexpected-ready':
        if negative == 'timeout':
            await asyncio.Event().wait()
        message = ('device kernel image is invalid' if negative == 'unrelated-error' else
                   "llama_init_from_model: context type MTP requested but model doesn't contain MTP layers")
        print(message, file=sys.stderr, flush=True)
        return 23
    print("E CUDA: no CUDA-capable device", file=sys.stderr, flush=True)
    if MODE == 'memory-full-offload':
        print('load_tensors: offloaded 41/41 layers to GPU', flush=True)
        print('load_tensors: CPU_Mapped model buffer size = 2.00 MiB', flush=True)
        print('load_tensors: Vulkan0 model buffer size = 10.00 MiB', flush=True)
        print('llama_kv_cache: Vulkan0 KV buffer size = 1.00 MiB', flush=True)
        print('llama_memory_recurrent: Vulkan0 RS buffer size = 2.00 MiB', flush=True)
        print('llama_context: Vulkan_Host output buffer size = 0.50 MiB', flush=True)
        print('sched_reserve: Vulkan0 compute buffer size = 3.00 MiB', flush=True)
        print('sched_reserve: Vulkan_Host compute buffer size = 1.00 MiB', flush=True)
    else:
        print("load_tensors: CPU model buffer size = 10.00 MiB", flush=True)
    if os.environ.get('STUB_FIM') in {'codegemma-split', 'codegemma-bad-piece'}:
        print('print_info: n_vocab = 71', flush=True)
    if os.environ.get('STUB_FIM') not in (None, 'negative', 'missing-route', 'codestral'):
        for kind, piece in fim_selected().items():
            print(f"llama_vocab: FIM {kind} token = {FIM_TOKENS[piece]} '{piece}'", flush=True)
        if os.environ['STUB_FIM'] in {'repo', 'qwen'}:
            for kind, piece in [('REP', '[REP]'), ('SEP', '[SEP]')]:
                print(f"llama_vocab: FIM {kind} token = {FIM_TOKENS[piece]} '{piece}'", flush=True)
    if MODE not in {'memory-missing', 'memory-full-offload'}:
        print("llama_kv_cache: CPU KV buffer size = 1.00 MiB", flush=True)
        print("llama_memory_recurrent: CPU RS buffer size = 2.00 MiB", flush=True)
        print("llama_context: CPU output buffer size = 0.50 MiB", flush=True)
        print("sched_reserve: CPU compute buffer size = 3.00 MiB", flush=True)
    if MODE in ("flood", "long-log"):
        for channel in (sys.stdout, sys.stderr):
            if MODE == "long-log":
                print("ě" * 100000, file=channel, flush=True)
            else:
                for i in range(1500):
                    print(f"{i}: " + "buffer diagnostic " * 12, file=channel)
                channel.flush()
    if MODE == "key-log":
        print("private key " + KEY, flush=True)
    if os.environ.get("STUB_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    loader = 'load_model_impl' if os.environ.get('STUB_BUILD') == 'b11552-d1570abd3d' else 'load_model'
    print(f"srv  {loader[:12]:>12}: initializing, n_slots = 1, n_ctx_slot = 2048, kv_unified = 'true'", flush=True)
    server = await asyncio.start_server(handle, "127.0.0.1", int(arg("--port")))
    port = server.sockets[0].getsockname()[1]
    if MODE != "hang-load":
        print(f"srv   run_server: listening on http://127.0.0.1:{port}", flush=True)
    else:
        print("Loading model", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
