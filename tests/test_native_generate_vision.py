"""Image generate keeps its public format while using the owned vision chat path."""

import asyncio
import json

import pytest

from test_native_manager import native_manager, records, wait_until
from test_native_public_api import public, stream_mode
from test_native_vision import vision
from vision_fixtures import RED_BLUE, BLUE_RED, RED_BLUE_URL, BLUE_RED_URL


def body(name, **extra):
    return {"model": name, "prompt": "Describe the panels.", "images": [RED_BLUE],
            "stream": False, "think": False, **extra}


def inference(manager):
    return [r for r in records(manager.test.roots[0], "requests")
            if r["path"] in {"/completion", "/v1/chat/completions", "/tokenize"}]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("image,expected", [(RED_BLUE, {"left": "red", "right": "blue"}),
                                           (BLUE_RED, {"left": "blue", "right": "red"})])
async def test_generate_image_result_schema_stats_and_native_payload(vision, stream, image, expected):
    manager, client, name, _ = vision
    schema = {"type": "object", "properties": {"left": {"type": "string"}, "right": {"type": "string"}}}
    response = await client.post("/api/generate", json=body(name, stream=stream, images=[image],
        format=schema, options={"temperature": 0, "num_predict": 48, "stop": ["CLIENT STOP"]}))
    assert response.status_code == 200
    frames = [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]
    assert all("message" not in frame and "choices" not in frame and "error" not in frame for frame in frames)
    assert json.loads("".join(frame["response"] for frame in frames)) == expected
    assert sum(frame["done"] for frame in frames) == 1 and frames[-1]["done_reason"] == "stop"
    assert frames[-1]["model"] == name and frames[-1]["context"] == []
    assert frames[-1]["prompt_eval_count"] == 6 and frames[-1]["prompt_eval_cached_count"] == 19
    assert frames[-1]["eval_count"] == 10 and frames[-1]["eval_duration"] == 200_100_000
    [call] = inference(manager)
    assert call["path"] == "/v1/chat/completions"
    sent = call["body"]
    assert sent["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert sent["messages"][1]["content"][0] == {"type": "text", "text": "Describe the panels."}
    assert sent["messages"][1]["content"][1]["image_url"]["url"] == (RED_BLUE_URL if image == RED_BLUE else BLUE_RED_URL)
    assert sent["response_format"] == {"type": "json_object", "schema": schema}
    assert sent["stop"] == ["CLIENT STOP"] and sent["temperature"] == 0 and sent["n_predict"] == 48
    assert sent["reasoning_effort"] == "none" and sent["reasoning_format"] == "deepseek"
    assert manager.native_snapshot()[0]["leases"] == 0 and len(manager.test.ledger) == 1


@pytest.mark.parametrize("system", [None, "", "REQUEST SYSTEM"])
async def test_system_precedence_multiple_images_and_json_format(vision, system):
    manager, client, name, _ = vision
    response = await client.post("/api/generate", json=body(name, system=system,
        images=[RED_BLUE, BLUE_RED], format="json"))
    assert response.status_code == 200
    sent = inference(manager)[0]["body"]
    assert sent["messages"][0] == {"role": "system", "content": "SYSTEM" if system is None else system}
    parts = sent["messages"][-1]["content"]
    assert [part["type"] for part in parts] == ["text", "image_url", "image_url"]
    assert [part["image_url"]["url"] for part in parts[1:]] == [RED_BLUE_URL, BLUE_RED_URL]
    assert sent["stop"] == ["TOML STOP"] and sent["temperature"] == 0.4
    assert sent["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("think,effort", [(True, None), (False, "none"), ("high", "high")])
async def test_generate_vision_preserves_structured_reasoning(vision, think, effort):
    manager, client, name, _ = vision
    response = await client.post("/api/generate", json=body(name, think=think, stream=True))
    frames = [json.loads(line) for line in response.text.splitlines()]
    assert "".join(frame.get("thinking", "") for frame in frames) == "myšlenka"
    sent = inference(manager)[0]["body"]
    assert sent.get("reasoning_effort") == effort
    assert sent["chat_template_kwargs"]["enable_thinking"] is (think is not False)


@pytest.mark.parametrize("prompt", [None, ""])
@pytest.mark.parametrize("stream", [False, True])
async def test_image_only_is_inference_and_keep_alive_zero_retires_after_reply(vision, prompt, stream):
    manager, client, name, _ = vision
    response = await client.post("/api/generate", json=body(name, prompt=prompt, stream=stream, keep_alive=0))
    frames = [json.loads(line) for line in response.text.splitlines()] if stream else [response.json()]
    assert response.status_code == 200 and frames[-1]["done_reason"] == "stop"
    assert json.loads("".join(frame["response"] for frame in frames)) == {"left": "red", "right": "blue"}
    assert len(inference(manager)) == 1
    await wait_until(lambda: not manager.native_snapshot())
    worker = manager.test.ledger[0]
    assert worker.process.returncode is not None and not worker.runtime_directory.exists()


@pytest.mark.parametrize("extra,status", [({"raw": True}, 400), ({"template": "{{ Prompt }}"}, 400),
    ({"template": ""}, 400), ({"suffix": "tail"}, 400), ({"context": [1]}, 400),
    ({"format": "invalid"}, 400)])
async def test_unsupported_combinations_fail_before_model_load(public, extra, status):
    manager, client, name, _ = public
    response = await client.post("/api/generate", json=body(name, **extra))
    assert response.status_code == status and "error" in response.json()
    assert not manager.test.ledger


@pytest.mark.parametrize("image", ["http://127.0.0.1:9/x", "file:///etc/passwd", "invalid!!",
    "data:text/plain;base64," + RED_BLUE, "data:image/png;base64,aGVsbG8="])
async def test_invalid_images_fail_before_load_even_with_empty_prompt(public, image):
    manager, client, name, _ = public
    response = await client.post("/api/generate", json=body(name, prompt="", images=[image], keep_alive=0))
    assert response.status_code == 400 and not manager.test.ledger


@pytest.mark.parametrize("stream", [False, True])
async def test_missing_live_vision_is_preheader_400_without_inference(public, stream):
    manager, client, name, _ = public
    response = await client.post("/api/generate", json=body(name, stream=stream))
    assert response.status_code == 400 and "does not support multimodal" in response.json()["error"]
    assert inference(manager) == [] and manager.native_snapshot()[0]["leases"] == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_decode_error_has_no_generate_success_terminal(vision, stream):
    manager, client, name, _ = vision
    manager.test.options["mode"] = "bad-image"
    response = await client.post("/api/generate", json=body(name, stream=stream))
    assert response.status_code == (200 if stream else 400)
    assert "Image decoder rejected" in response.text and '"done": true' not in response.text
    assert manager.native_snapshot()[0]["leases"] == 0


async def test_unload_preserves_active_generate_image_stream_until_completion(vision):
    manager, client, name, _ = vision
    stream_mode(manager, "controlled")
    task = asyncio.create_task(client.post("/api/generate", json=body(name, stream=True)))
    try:
        await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / "started").exists())
        generation = manager.native_snapshot()[0]["generation"]
        await manager.unload_native_model(name)
        [process] = (await client.get("/api/ps")).json()["models"]
        assert process["state"] == "draining" and process["leases"] == 1
    finally:
        (manager.test.roots[0] / "release").touch()
        response = await task
    frames = [json.loads(line) for line in response.text.splitlines()]
    assert frames[-1]["done"] and frames[-1]["done_reason"] == "stop"
    await manager.wait_native_closed(generation)
    assert (await client.get("/api/ps")).json() == {"models": []}


@pytest.mark.parametrize("stream", [False, True])
async def test_cancelled_generate_image_releases_upstream_and_reuses_worker(vision, stream):
    manager, client, name, _ = vision
    original_chats = []
    if stream:
        stream_mode(manager, "hold")
    else:
        original = manager._native._factory
        def factory(spec):
            worker = original(spec)
            chat = worker.chat
            original_chats.append(chat)
            worker.chat = lambda payload: chat({**payload, "stub": "hold"})
            return worker
        manager._native._factory = factory
    task = asyncio.create_task(client.post("/api/generate", json=body(name, stream=stream)))
    await wait_until(lambda: manager.test.roots and (manager.test.roots[0] / "started").exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: (manager.test.roots[0] / "cancelled").exists())
    assert manager.native_snapshot()[0]["leases"] == 0 and manager.test.ledger[0].process.returncode is None
    if not stream:
        manager.test.ledger[0].chat = original_chats[0]
    response = await client.post("/api/generate", json=body(name, images=[BLUE_RED]))
    assert response.status_code == 200 and len(manager.test.ledger) == 1
    assert json.loads(response.json()["response"]) == {"left": "blue", "right": "red"}
