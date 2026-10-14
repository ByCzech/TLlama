# Native text API adapter checkpoint (05a)

This checkpoint adds adapters/tests. Public routers, request schema fields and
the application engine choice remain unchanged. The protocol remains pinned to
b11552-ec2797836f. Public switching belongs to 05b, vision/memory to 06 and final
binding removal to 07.

## Request preparation

`native_requests.prepare_ollama_chat()` and `prepare_openai_chat()` consume the
existing request schemas. Their shared `prepare_chat()` returns an immutable
NativeRequest; each body() produces fresh JSON containers. Pass the acquired
lease's resolved configuration rather than a newly read TOML.

Sampling keeps request > TOML > global > supplied GGUF recommendations > TLlama
baseline priority. System content comes from the resolved TOML snapshot; an
explicit system/developer message, including empty content, suppresses it.
Names, tool-call IDs/arguments, text parts and assistant reasoning history are
retained. Caller-supplied GGUF header metadata must belong to the same model
version; the adapter does not read headers, render templates or load models.

Think aliases become typed chat_template_kwargs and native reasoning_effort
where specified. Unspecified think does not override template defaults. OpenAI
top-level effort wins over nested reasoning.effort. Chat requests explicitly ask
for reasoning_format=deepseek, the native structured output parser setting;
this is not a requirement for a DeepSeek model or a request to enable thinking.

Ollama JSON/schema and OpenAI response_format map to native JSON fields without
Python grammar objects. This validates the API shape, not every JSON Schema
feature or actual constrained generation; the real-runtime checkpoint must
test those. prepare_completion() receives exact raw/already-rendered text and
maps format to json_schema. It does not select FIM, append suffixes, interpret
generate.context or invent per-request template behavior.

The pinned tool_choice parser accepts only auto/none/required. A modern named
function choice selects exactly its original declared tool plus required;
missing/ambiguous names fail before transport. Ollama tool-call history gains
OpenAI's function type discriminator where omitted. This is wire mapping,
not new application tool orchestration.

Media/images and deprecated functions/function_call inputs/history are explicitly
refused instead of dropped. Neither refusal currently affects a public route.
Decide deprecated function compatibility before the 05b switch; vision is 06.

## Ownership and neutral results

`native_api.NativeAPI(worker, request)` uses the existing worker transport.
result() returns a NativeResult. stream() is an async context manager yielding
neutral deltas followed by one terminal result. It owns its upstream response
context, not model residency, and adds no queue, load, retry or inference replay.

```python
async with manager.acquire_native_model(request.model) as lease:
    prepared = prepare_ollama_chat(request, resolved=lease.resolved)
    result = await NativeAPI(lease.worker, prepared).result()
    payload = ollama_response(result, ReplyContext(request.model))
```

For streams, nest `async with api.stream() as events` inside the lease and
iterate ollama_stream(events, context) or openai_stream(events, context) there.
The lease must cover iteration, slot wait, early exit and cancellation, not just
creation of an iterator. Leaving the upstream context closes its HTTP response.
CancelledError propagates instead of becoming a public inference error.

Chat content/reasoning_content are distinct native fields; literal think markers
in content are not reparsed. Raw completion has no structured channel, so its
caller-selected ReasoningStreamSplitter handles both full text and stream
deltas, preserving whitespace equally. Reasoning-only length results remain
successful limited replies, with an empty answer and separate reasoning.

## Statistics and public formats

Statistics belong to one result/stream. No shared counter reset or tokenizer
estimate is used. The distinctions below are deliberate:

| Value | Pinned source | Public mapping |
|---|---|---|
| Entire input, including reused prefix | usage.prompt_tokens; raw tokens_evaluated | OpenAI prompt_tokens |
| Newly evaluated input | timings.prompt_n | Ollama prompt_eval_count |
| Reused input prefix | usage.prompt_tokens_details.cached_tokens or timings.cache_n | OpenAI cached_tokens; Ollama prompt_eval_cached_count |
| Generated tokens, including reasoning | usage.completion_tokens; raw tokens_predicted; timings.predicted_n | OpenAI completion_tokens; Ollama eval_count |
| Prompt evaluation time | timings.prompt_ms | Ollama prompt_eval_duration, ms to ns |
| Generation time | timings.predicted_ms | Ollama eval_duration, ms to ns |

An input of 25 with 21 cached and 4 evaluated is OpenAI input count 25; Ollama
pairs the measured evaluation duration with count 4 and cached count 21.
Raw tokens_cached is slot occupancy AFTER generation, never measured input reuse.
Unknown measurements are omitted; measured zero stays zero. Missing counts do
not become a fabricated total. Invalid types, negative/nonfinite values and
inconsistent OpenAI totals are protocol errors.

ReplyContext keeps one public reference, unique ID and created timestamp across
OpenAI chunks. Optional monotonic request-start/lease-acquired timestamps yield
total_duration and load_duration. The latter preserves lookup/admission timing,
including capacity wait; transport-slot wait stays in total_duration. Native
prompt/generation times are not queue wait, TTFT or cold worker load duration.
Those extra measurements are not invented.

Ollama uses NDJSON with message.thinking or generate.thinking. OpenAI uses SSE
with message/delta.reasoning_content and answer-only content. Reasoning is not
merged into content or hidden merely because think=False; think controls the
template. Modern tool fields/deltas are preserved as received, without claiming
complete Ollama tool reassembly or SDK parity.

Final Ollama chat lines carry the required empty assistant message. Raw generate
keeps the empty context compatibility placeholder, not KV/conversation history.
Native eos/word maps to stop; limit maps to length. Chat usage may follow finish
in empty choices. Public success is emitted only after transport verifies DONE
or raw stop. Invalid JSON, premature EOF or native errors after partial output
produce a public error envelope without successful done/DONE. Optional public
OpenAI usage has empty choices and precedes DONE; the worker always requests
native usage internally regardless of that public inclusion policy.

public_error() prepares pre-header status/envelopes, preserving existing load
error differences, broken-TOML 500 and native request errors. Stream errors use
the existing flat Ollama/structured OpenAI envelopes. Final routing/status
application remains for 05b.

## Verification and remaining boundary

125 focused tests cover request snapshots, precedence, format/think/tool mapping,
structured/raw reasoning, measured statistics, malformed protocol and terminal
order/error envelopes. Real loopback subprocess tests cover both result/stream
paths, fragmented SSE, crash without replay, early close and cancellation with
subsequent reuse of the same worker. No model weights are loaded.

No hardware run is needed for 05a. Before 05b, agree generate template/suffix
policy, deprecated function mapping and the vision/ps boundary. Its real text
API smoke must test reasoning control, constrained generation and public client
shapes; the binding/application cleanup is still a separate milestone.
