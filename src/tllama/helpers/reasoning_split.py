from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReasoningFormat = Literal[
    "none",
    "granite",
    "qwen_like",
    "gemma_channel",
    "gpt_oss_channel",
    "deepseek"
]


def _normalize_text(value) -> str:
    return "" if value is None else str(value)


def _think_explicitly_disabled(think_value) -> bool:
    if think_value is False:
        return True
    if isinstance(think_value, str) and think_value.strip().lower() == "none":
        return True
    return False


def detect_reasoning_format(model_name: str, metadata_info: dict | None = None) -> ReasoningFormat:
    model_lower = _normalize_text(model_name).lower()
    template = _normalize_text((metadata_info or {}).get("template")).lower()

    if "gpt-oss" in model_lower or (
        "<|channel|>analysis<|message|>" in template
        and "<|channel|>final<|message|>" in template
    ):
        return "gpt_oss_channel"

    if "gemma" in model_lower or (
        "<|channel>thought" in template and "<channel|>" in template
    ):
        return "gemma_channel"

    if "granite" in model_lower or (
        "<response>" in template and "<think>" in template
    ):
        return "granite"

    if "deepseek" in model_lower:
        return "deepseek"

    if "qwen" in model_lower:
        return "qwen_like"

    return "none"


@dataclass
class _FormatSpec:
    open_marker: str | None
    close_marker: str | None
    response_open_marker: str | None = None
    response_close_marker: str | None = None
    qwen_like: bool = False


class ReasoningStreamSplitter:
    def __init__(self, fmt: ReasoningFormat, think_value=None):
        self.fmt = fmt
        self.think_value = think_value
        self.buffer = ""

        self.spec = self._build_spec(fmt)

        if fmt == "qwen_like" and not _think_explicitly_disabled(think_value):
            self.state = "thinking"
        else:
            self.state = "response"

        if fmt == "granite":
            self.after_thinking_state = "seek_response"
        else:
            self.after_thinking_state = "response"

    def _build_spec(self, fmt: ReasoningFormat) -> _FormatSpec:
        if fmt == "granite":
            return _FormatSpec(
                open_marker="<think>",
                close_marker="</think>",
                response_open_marker="<response>",
                response_close_marker="</response>",
            )
        if fmt == "qwen_like":
            return _FormatSpec(
                open_marker="<think>",
                close_marker="</think>",
                qwen_like=True,
            )
        if fmt == "deepseek":
            return _FormatSpec(
                open_marker="<think>",
                close_marker="</think>",
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
        if text:
            self.buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
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

            # qwen-like special mode:
            # if reasoning is not explicitly disabled, response starts in thinking
            # and only switches after </think>.
            if self.spec.qwen_like:
                if self.state == "thinking":
                    if self.spec.open_marker and self.buffer.startswith(self.spec.open_marker):
                        self.buffer = self.buffer[len(self.spec.open_marker):]
                        continue

                    close_marker = self.spec.close_marker
                    idx = self.buffer.find(close_marker)
                    if idx != -1:
                        out.extend(self._emit("thinking", self.buffer[:idx]))
                        self.buffer = self.buffer[idx + len(close_marker):]
                        self.state = "response"
                        continue

                    out.extend(self._flush_plain("thinking", final, [close_marker, self.spec.open_marker]))
                    break

                # response mode
                if self.spec.open_marker:
                    idx = self.buffer.find(self.spec.open_marker)
                    if idx != -1:
                        out.extend(self._emit("response", self.buffer[:idx]))
                        self.buffer = self.buffer[idx + len(self.spec.open_marker):]
                        self.state = "thinking"
                        continue

                out.extend(self._flush_plain("response", final, [self.spec.open_marker]))
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

            # generic open/close formats: deepseek, gemma_channel, gpt_oss_channel
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
                    self.state = self.after_thinking_state
                    continue

                out.extend(self._flush_plain("thinking", final, [self.spec.close_marker]))
                break

            break

        return out


def split_full_text_by_reasoning_format(
    full_text: str,
    fmt: ReasoningFormat,
    think_value=None,
) -> tuple[str, str]:
    full_text = full_text or ""

    if fmt == "qwen_like":
        stripped = full_text.strip()

        # 1) Standard tagged form: <think> ... </think> answer
        if "<think>" in stripped and "</think>" in stripped:
            left, right = stripped.split("</think>", 1)
            left = left.split("<think>", 1)[-1]
            return left.strip(), right.strip()

        # 2) Qwen-like broken form: reasoning ... </think> answer
        #    (missing starting <think>, but ending are there)
        if "</think>" in stripped:
            left, right = stripped.split("</think>", 1)
            return left.strip(), right.strip()

        # 3) If thinking is explicitly disabled, without end marker we don't guess nothing
        think_disabled = (
            think_value is False
            or (
                isinstance(think_value, str)
                and think_value.strip().lower() in {"false", "none"}
            )
        )
        if think_disabled:
            return "", stripped

        # 4) If thinking is enabled and text seems like reasoning-only output,
        #    we getting it whole as thinking
        lowered = stripped.lower()
        if (
            lowered.startswith("thinking process:")
            or lowered.startswith("here's a thinking process:")
            or lowered.startswith("here is a thinking process:")
        ):
            return stripped, ""

    splitter = ReasoningStreamSplitter(fmt, think_value=think_value)

    parts: list[tuple[str, str]] = []
    parts.extend(splitter.push(full_text))
    parts.extend(splitter.finish())

    thinking_parts: list[str] = []
    response_parts: list[str] = []

    for kind, piece in parts:
        if kind == "thinking":
            thinking_parts.append(piece)
        else:
            response_parts.append(piece)

    return "".join(thinking_parts).strip(), "".join(response_parts).strip()
