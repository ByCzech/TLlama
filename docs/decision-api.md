# Public text Decision API

`POST /v1/systemone` exposes the accepted text Decision contract through managed
TLlama workers. It is a native-compatible **extension**, not a standard OpenAI
Chat Completions or Ollama method. Use HTTP JSON; no high-level SDK method is
claimed. Native/managed/public comparisons passed for Laya, Nimble and Clef
on native 0.7.0, MI50/Vulkan, F16 K/V and FA off; see
[runtime-audit-070.md](runtime-audit-070.md). Keep the independent references.

Run the [0069 native / managed / public comparison](../scripts/decision-api-smoke.md)
to verify the managed launch and public text adapter on those existing models.

```bash
curl http://127.0.0.1:54800/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ggml-org/Laya-GGUF/Laya-Q8_0",
    "state": {"ticket": "I was charged twice. Please refund the duplicate payment."},
    "questions": {
      "team": {"type": "choice", "instructions": "Which department should handle this?",
               "criteria": {"billing": "payments and refunds", "shipping": "parcels"}},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["can wait", "today", "right now"]},
      "angry": {"type": "noul", "instructions": "Is the customer angry?"}
    }
  }'
```

Only `model`, `state` and `questions` are accepted. Model must be a nonblank
registered reference. State and instructions must be present and non-null;
otherwise their finite JSON values are preserved, including false,0,empty
strings,arrays and nested objects. Option descriptions and score levels may
also be arbitrary finite JSON. Question IDs and choice keys are literal strings.
There is no de-escaping or generated answer-text parsing in this adapter.

| Question type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Object of1..255 option keys and JSON descriptions | `choice`, keyed `probabilities`, `confidence` |
| `score` | Ordered array of2..10 levels | Expected zero-based `score`, `legend`, keyed `probabilities`, `confidence` |
| `noul` | Object or null/omitted; optional `true`/`false` descriptions | Probability of true in `noul` |

Each answer includes its `type`. Response question IDs retain request order.
Laya/Nimble choice probabilities retain option request order; Clef returns them
in native sorted-key order. Score probabilities/legend use keys `"0"`, `"1"`,etc.
Native owns calibrated probabilities,expected scores and confidence formulas;
the worker verifies them and returns the values without repair/normalization.

The successful response has exactly `model`, `answers` and `usage`. Model is the
admitted public reference rather than the worker's private alias. Answers and
usage are the validated native objects,without synthetic timings or token counts.
`usage.input_tokens` is a strict positive integer; `output_tokens` is integer0
because no answer text is generated. Native Nimble whole-schema field tasks and
Clef joint tasks have different accounting: standalone counts need not add to a
combined request. Changing the schema/order can change probabilities too.

## Validation and limits

Requests are checked before model resolution/admission. The wire body is read
incrementally with a2MiB limit before JSON decoding. UTF-8 JSON must be an object;
malformed JSON,duplicate keys at any nesting,NaN/Infinity,non-string object keys,
depth over32 or more than100000 value nodes are rejected. The existing private
2MiB serialized JSON limit also applies. At most32 questions are accepted.

Images,audio,video,remote media,streaming,sampling/template/system fields,
`options`,`runner`,`keep_alive` and unknown question fields are unsupported and
refused. A URL inside ordinary state text is just text; the adapter does not
fetch it. The native parser also recognizes media parts in a state that is an
array of chat messages or an object with a `messages` array. In those two layouts,
`image_url`/`input_audio` parts carrying their corresponding data fields are
refused before load. Similarly shaped data elsewhere remains literal JSON.
There is no preload/unload sentinel in this inference endpoint: null
state or empty questions is an invalid request. Ordinary empty Ollama generate/
chat load controls remain available; for example `{"model":"...","keep_alive":0}`
at `/api/generate` unloads through the existing protected lifecycle.

Device,context,batch/ubatch,K/V,FA and TTL use the model/server configuration.
There are no request capacity overrides or silently applied diagnostic settings.
The model's embedded Decision template/head performs tokenization and any trained
head truncation. This public layer does not fake a text-only token estimate or
truncate state to make a request fit. See [native-decision.md](native-decision.md)
for accepted metadata,conflicting settings and the required native runtime.

`/api/show` advertises only `decision` for an integrated valid Decision header
without configured semantic embedding mode or projector. This is an offline
model capability,not a startup or quality guarantee; conflicting generation
settings can still fail configuration validation. Incomplete/unintegrated
declared Decision heads are not advertised as generation or Decision. No media
capability is claimed. OpenAI `/v1/models` keeps its existing list format.

## Errors and ownership

Errors use the existing `/v1` envelope:

```json
{"error":{"message":"...","type":"invalid_request_error","param":null,"code":null}}
```

| Status | Meaning |
| --- | --- |
|400| Invalid JSON/contract,bounds,wrong model operation or incompatible load settings; native400 is retained |
|404| Registered model/weight not found |
|413| Raw HTTP body exceeds2MiB |
|415| Explicit unsupported content type (existing JSON middleware still handles undeclared CLI bodies) |
|500| Bad TOML,worker startup/runtime/transport/protocol failure; native500 is retained |
|503| Manager shutting down |

Other native error statuses retain the existing worker/public mapping. No partial
answers are returned after a failure and a failed request is never replayed.
The next independent request may reuse a healthy worker or reload an exited one.

The public request holds its lease while waiting for the shared native slot and
during inference. Client task cancellation or real HTTP disconnect cancels owned
work,closes the native response and releases the slot/lease. Unload cannot kill
an already admitted lease. Native early compute stop and a cancellation latency
bound remain unverified. No second manager/queue or native code change is added.

```bash
pytest -q tests/test_decision_api.py tests/test_native_decision.py \
  tests/test_decision_smoke.py tests/test_native_public_api.py \
  tests/test_embedding_api.py tests/test_embedding_media_api.py
pytest -q
```

Synthetic GGUF/TCP tests validate all three types against direct managed native
responses,literal JSON,limits/errors,corruption recovery,wrong-mode admission,
crash without replay,queued cancellation,real TCP disconnect and owned cleanup.
They do not certify model quality or replace the separately reviewed hardware
comparison.
