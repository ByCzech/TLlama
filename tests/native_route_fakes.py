"""Controlled native leases for public contract tests; no binding-shaped fakes."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tllama.config import BackendConfig
from tllama.helpers.model_toml import VirtualModelSpec
from tllama.native_config import resolve_worker_config
from tllama.residency import ModelDefinition, ModelLease


class ReplyWorker:
    def __init__(self, metadata, *, content="ok", reasoning="", usage=None, timings=None, vision=False):
        self.props = {"chat_template": metadata.get("template", "{{ Prompt }}"), "bos_token": "", "eos_token": ""}
        self.props['modalities'] = {'vision':vision}
        self.content, self.reasoning = content, reasoning
        self.usage = {"prompt_tokens": 15, "completion_tokens": 542, "total_tokens": 557} if usage is None else usage
        self.timings = {"prompt_n": 15, "predicted_n": 542, "prompt_ms": 172.9, "predicted_ms": 5192.2} if timings is None else timings
        self.error = None
        self.calls, self.prompts = [], []
        self.slot = asyncio.Lock()
        self.before_completion = None
        self.after_completion = None

    async def tokenize(self, content, *, add_special=True):
        self.prompts.append(content)
        return [ord(c) for c in content]

    async def _reply(self, kind, payload):
        self.calls.append((kind, deepcopy(payload)))
        if self.before_completion:
            await self.before_completion(kind, payload)
        if self.error:
            raise self.error
        if kind == "chat":
            result = {"choices": [{"message": {"content": self.content, "reasoning_content": self.reasoning},
                                    "finish_reason": "stop"}], "usage": deepcopy(self.usage)}
        else:
            result = {"content": self.content, "stop": True, "stop_type": "eos"}
            for source, target in [("prompt_tokens", "tokens_evaluated"), ("completion_tokens", "tokens_predicted")]:
                if source in self.usage:
                    result[target] = self.usage[source]
        result["timings"] = deepcopy(self.timings)
        if self.after_completion:
            self.after_completion(kind, payload)
        return result

    async def chat(self, payload):
        async with self.slot:
            return await self._reply("chat", payload)

    async def completion(self, payload):
        async with self.slot:
            return await self._reply("completion", payload)

    @asynccontextmanager
    async def _stream(self, kind, payload):
        async with self.slot:
            result = await self._reply(kind, payload)

            async def events():
                if kind == "chat":
                    yield {"choices": [{"delta": {"content": self.content, "reasoning_content": self.reasoning}}]}
                    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    yield {"choices": [], "usage": result["usage"], "timings": result["timings"]}
                else:
                    yield result

            iterator = events()
            try:
                yield iterator
            finally:
                await iterator.aclose()

    def chat_stream(self, payload):
        return self._stream("chat", payload)

    def completion_stream(self, payload):
        return self._stream("completion", payload)


def install_reply(monkeypatch, manager, *, metadata=None, **values):
    from tllama.routers import ollama, openai
    metadata = deepcopy(metadata or {})
    worker = ReplyWorker(metadata, **values)
    state = SimpleNamespace(worker=worker, manager=manager, refs=0, actions=[], load_delay=0, load_error=None)

    @asynccontextmanager
    async def acquire(name, num_ctx=None, keep_alive=None):
        if state.load_delay:
            await asyncio.sleep(state.load_delay)
        if state.load_error:
            raise state.load_error
        spec = VirtualModelSpec(llm_model="Local/m.gguf", template=metadata.get("template"),
            system_prompt=metadata.get("default_system_prompt"), sampling=metadata.get("sampling_defaults", {}),
            stop=metadata.get("stop_defaults", []))
        resolved = resolve_worker_config(BackendConfig(runtime_overrides={"device": "none"},
            sampling_overrides=manager.config.sampling_overrides), Path('/models/m.gguf'), spec, num_ctx=num_ctx)
        state.refs += 1
        try:
            yield ModelLease(1, ModelDefinition(name, resolved, metadata=metadata), worker)
        finally:
            state.refs -= 1
            if manager.resolve_keep_alive(keep_alive) == 0:
                state.actions.append(("unload", name))

    async def preload(name, **kwargs):
        state.actions.append(("load", name))
        if state.load_error:
            raise state.load_error
        return 1

    async def unload(name):
        state.actions.append(("unload", name))

    monkeypatch.setattr(manager, "acquire_native_model", acquire)
    monkeypatch.setattr(manager, "preload_native_model", preload)
    monkeypatch.setattr(manager, "unload_native_model", unload)
    monkeypatch.setattr(ollama, "model_manager", manager)
    monkeypatch.setattr(openai, "model_manager", manager)
    return state
