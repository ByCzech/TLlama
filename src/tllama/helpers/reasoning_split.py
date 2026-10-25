from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReasoningFormat = Literal[
    "none",
    "granite",
    "think_prefill",
    "gemma_channel",
    "gpt_oss_channel",
    "think_prefix",
]


def detect_reasoning_format(metadata_info: dict | None = None) -> ReasoningFormat:
    """Select a format from the effective template and evidenced GGUF contract.

    Architecture alone does not establish thinking for general families such
    as Qwen, Gemma or Granite. In particular, no public model alias is consulted.
    Marker spelling is case-sensitive, just like the generated protocol.
    """
    metadata = metadata_info or {}
    template = metadata.get("template") or ""

    if ("<|channel|>analysis<|message|>" in template
            and "<|channel|>final<|message|>" in template):
        return "gpt_oss_channel"
    if "<|channel>thought" in template and "<channel|>" in template:
        return "gemma_channel"
    if "<response>" in template and "<think>" in template and "</think>" in template:
        return "granite"

    # The observed Instella-MoE-Think contract emits explicit tags although
    # its original template only renders message.content. Other templates
    # establish this contract through the explicit pair, regardless of arch.
    if metadata.get("arch") == "instella-moe" or (
        "<think>" in template and "</think>" in template
    ):
        return "think_prefix"
    return "none"


def completion_reasoning_format(metadata_info: dict | None, generation_prefix: str | None) -> ReasoningFormat:
    """Implicit tag thinking requires an actual, unclosed generation prefill.

    Non-raw callers supply only the suffix appended by add_generation_prompt,
    not the user text. Raw callers explicitly own their whole prompt prefix.
    An empty completed think block, including think=false templates, is not an
    open reasoning channel. Never infer this mode merely from architecture.
    """
    fmt = detect_reasoning_format(metadata_info)
    if fmt == "think_prefix" and generation_prefix is not None and generation_prefix.rstrip().endswith("<think>"):
        return "think_prefill"
    return fmt


class ChatReasoningNormalizer:
    """Fallback for one explicit leading think block in a known tag format.

    Native structured reasoning wins. Unlike raw Qwen completion, chat content
    never starts in implicit thinking mode. After the first answer character,
    all markers are literal, including examples in the answer itself.
    """

    def __init__(self, fmt: ReasoningFormat, *, prefilled: bool = False):
        self.state = "prefix" if fmt in {"think_prefill", "think_prefix"} else "content"
        if prefilled:
            self.state = "prefill"
        self.buffer = ""

    def push(self, content: str, reasoning: str = "") -> list[tuple[str, str]]:
        if reasoning:
            pending = self.finish()
            self.state = "content"
            return pending + [("thinking", reasoning), ("response", content)]
        if self.state == "content":
            return [("response", content)] if content else []
        self.buffer += content
        out = []
        if self.state in {"prefix", "prefill"}:
            candidate = self.buffer.lstrip()
            marker = "<think>"
            if marker.startswith(candidate) and candidate != marker:
                return []
            if candidate.startswith(marker):
                whitespace = self.buffer[:len(self.buffer) - len(candidate)]
                if whitespace:
                    out.append(("thinking", whitespace))
                self.buffer = candidate[len(marker):]
            elif self.state == "prefix":
                return self.finish()
            self.state = "thinking"
        marker = "</think>"
        idx = self.buffer.find(marker)
        if idx >= 0:
            out += [("thinking", self.buffer[:idx]), ("response", self.buffer[idx + len(marker):])]
            self.buffer = ""
            self.state = "content"
        elif len(self.buffer) >= len(marker):
            # Retain a possible closing tag split across transport deltas.
            keep = len(marker) - 1
            out.append(("thinking", self.buffer[:-keep]))
            self.buffer = self.buffer[-keep:]
        return [(kind, text) for kind, text in out if text]

    def finish(self) -> list[tuple[str, str]]:
        kind = "thinking" if self.state in {"thinking", "prefill"} else "response"
        text, self.buffer = self.buffer, ""
        self.state = "content"
        return [(kind, text)] if text else []


@dataclass
class _FormatSpec:
    open_marker: str | None
    close_marker: str | None
    response_open_marker: str | None = None
    response_close_marker: str | None = None


class ReasoningStreamSplitter:
    def __init__(self, fmt: ReasoningFormat):
        self.fmt = fmt
        self.buffer = ""
        self.prefix_normalizer = (ChatReasoningNormalizer(fmt, prefilled=fmt == "think_prefill")
                                  if fmt in {"think_prefix", "think_prefill"} else None)

        self.spec = self._build_spec(fmt)

        self.state = "response"

    def _build_spec(self, fmt: ReasoningFormat) -> _FormatSpec:
        if fmt == "granite":
            return _FormatSpec(
                open_marker="<think>",
                close_marker="</think>",
                response_open_marker="<response>",
                response_close_marker="</response>",
            )
        if fmt == "gemma_channel":
            return _FormatSpec(
                open_marker="<|channel>thought\n",
                close_marker="<channel|>",
            )
        if fmt == "gpt_oss_channel":
            return _FormatSpec(
                open_marker="<|channel|>analysis<|message|>",
                close_marker="<|end|><|start|>assistant<|channel|>final<|message|>",
            )
        return _FormatSpec(open_marker=None, close_marker=None)

    def push(self, text: str) -> list[tuple[str, str]]:
        if self.prefix_normalizer is not None:
            return self.prefix_normalizer.push(text)
        if text:
            self.buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        if self.prefix_normalizer is not None:
            return self.prefix_normalizer.finish()
        return self._drain(final=True)

    def _emit(self, kind: str, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        return [(kind, text)]

    def _tail_len(self, markers: list[str | None]) -> int:
        lengths = [len(m) - 1 for m in markers if m]
        return max(lengths, default=0)

    def _flush_plain(self, kind: str, final: bool, markers: list[str | None]) -> list[tuple[str, str]]:
        if not self.buffer:
            return []

        if final:
            text = self.buffer
            self.buffer = ""
            return self._emit(kind, text)

        keep = self._tail_len(markers)
        if keep <= 0:
            text = self.buffer
            self.buffer = ""
            return self._emit(kind, text)

        if len(self.buffer) <= keep:
            return []

        text = self.buffer[:-keep]
        self.buffer = self.buffer[-keep:]
        return self._emit(kind, text)

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []

        while True:
            if not self.buffer:
                break

            # granite: <think>...</think><response>...</response>
            if self.fmt == "granite":
                if self.state == "response":
                    if self.spec.response_open_marker and self.buffer.startswith(self.spec.response_open_marker):
                        self.buffer = self.buffer[len(self.spec.response_open_marker):]
                        continue

                    if self.spec.open_marker:
                        idx = self.buffer.find(self.spec.open_marker)
                        if idx != -1:
                            out.extend(self._emit("response", self.buffer[:idx]))
                            self.buffer = self.buffer[idx + len(self.spec.open_marker):]
                            self.state = "thinking"
                            continue

                    if self.spec.response_close_marker:
                        idx = self.buffer.find(self.spec.response_close_marker)
                        if idx != -1:
                            out.extend(self._emit("response", self.buffer[:idx]))
                            self.buffer = self.buffer[idx + len(self.spec.response_close_marker):]
                            continue
                    out.extend(
                        self._flush_plain(
                            "response",
                            final,
                            [self.spec.open_marker, self.spec.response_open_marker, self.spec.response_close_marker],
                        )
                    )
                    break

                if self.state == "thinking":
                    if self.buffer.startswith(self.spec.open_marker):
                        self.buffer = self.buffer[len(self.spec.open_marker):]
                        continue

                    idx = self.buffer.find(self.spec.close_marker)
                    if idx != -1:
                        out.extend(self._emit("thinking", self.buffer[:idx]))
                        self.buffer = self.buffer[idx + len(self.spec.close_marker):]
                        self.state = "seek_response"
                        continue

                    out.extend(self._flush_plain("thinking", final, [self.spec.close_marker]))
                    break

                if self.state == "seek_response":
                    if self.spec.response_open_marker:
                        idx = self.buffer.find(self.spec.response_open_marker)
                        if idx != -1:
                            self.buffer = self.buffer[idx + len(self.spec.response_open_marker):]
                            self.state = "response"
                            continue

                    if final:
                        text = self.buffer
                        self.buffer = ""
                        if text.strip():
                            out.extend(self._emit("response", text))
                        break

                    break

            # Generic channel formats: gemma_channel, gpt_oss_channel
            if self.state == "response":
                if self.spec.open_marker:
                    idx = self.buffer.find(self.spec.open_marker)
                    if idx != -1:
                        out.extend(self._emit("response", self.buffer[:idx]))
                        self.buffer = self.buffer[idx + len(self.spec.open_marker):]
                        self.state = "thinking"
                        continue

                out.extend(self._flush_plain("response", final, [self.spec.open_marker]))
                break

            if self.state == "thinking":
                idx = self.buffer.find(self.spec.close_marker)
                if idx != -1:
                    out.extend(self._emit("thinking", self.buffer[:idx]))
                    self.buffer = self.buffer[idx + len(self.spec.close_marker):]
                    self.state = "response"
                    continue

                out.extend(self._flush_plain("thinking", final, [self.spec.close_marker]))
                break

            break

        return out
