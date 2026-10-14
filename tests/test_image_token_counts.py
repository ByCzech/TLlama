"""Native usage includes the entire input; no binding counter correction."""

CHAT = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "input"}]}


def test_ollama_reports_native_evaluated_input_count(client, native_reply):
    native_reply(usage={"prompt_tokens": 179, "completion_tokens": 1217}, timings={"prompt_n": 179})
    assert client.post('/api/chat', json=CHAT).json()["prompt_eval_count"] == 179


def test_ollama_reports_native_generated_count(client, native_reply):
    native_reply(usage={"prompt_tokens": 179, "completion_tokens": 1217}, timings={"prompt_n": 179})
    assert client.post('/api/chat', json=CHAT).json()["eval_count"] == 1217


def test_missing_evaluation_measurement_does_not_masquerade_as_logical_input(client, native_reply):
    native_reply(usage={"prompt_tokens": 179, "completion_tokens": 1217}, timings={})
    body = client.post('/api/chat', json=CHAT).json()
    assert "prompt_eval_count" not in body and body["eval_count"] == 1217


def test_openai_reports_full_native_input_including_cached_prefix(client, native_reply):
    native_reply(usage={"prompt_tokens": 179, "completion_tokens": 1217}, timings={"prompt_n": 39, "cache_n": 140})
    body = client.post('/v1/chat/completions', json=CHAT).json()
    assert body["usage"]["prompt_tokens"] == 179
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 140


def test_openai_total_is_the_known_native_input_and_output(client, native_reply):
    native_reply(usage={"prompt_tokens": 179, "completion_tokens": 1217}, timings={})
    usage = client.post('/v1/chat/completions', json=CHAT).json()["usage"]
    assert usage["total_tokens"] == 1396


def test_openai_usage_does_not_require_context_counters(client, native_reply):
    native_reply(usage={"prompt_tokens": 39, "completion_tokens": 1217}, timings={})
    assert client.post('/v1/chat/completions', json=CHAT).json()["usage"]["prompt_tokens"] == 39
