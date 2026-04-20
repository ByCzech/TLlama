import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class ReasoningFormat:
    start: str
    end: str
    source: str = "template"


def infer_reasoning_format_from_template(template: Optional[str]) -> Optional[ReasoningFormat]:
    """
    Tries to infer reasoning markers directly from the raw chat template.
    No model whitelist.
    """
    if not template:
        return None

    # XML/HTML-like tags, e.g. <think>...</think>, <thinking>...</thinking>
    for m in re.finditer(
        r"<(?P<tag>[A-Za-z0-9:_-]*(?:think|thinking|reason|reasoning|analysis)[A-Za-z0-9:_-]*)>",
        template,
        flags=re.IGNORECASE,
    ):
        tag = m.group("tag")
        start = m.group(0)
        end = f"</{tag}>"
        if end in template:
            return ReasoningFormat(start=start, end=end, source="template")

    # BBCode-like tags, e.g. [think]...[/think]
    for m in re.finditer(
        r"\[(?P<tag>[A-Za-z0-9:_-]*(?:think|thinking|reason|reasoning|analysis)[A-Za-z0-9:_-]*)\]",
        template,
        flags=re.IGNORECASE,
    ):
        tag = m.group("tag")
        start = m.group(0)
        end = f"[/{tag}]"
        if end in template:
            return ReasoningFormat(start=start, end=end, source="template")

    return None


def detect_reasoning_format(model_name: str, metadata_info: Optional[Dict[str, Any]] = None) -> Optional[ReasoningFormat]:
    """
    Primary strategy: infer from template/metadata.
    Unknown => None => passthrough mode.
    """
    metadata_info = metadata_info or {}
    template = metadata_info.get("template") or ""
    return infer_reasoning_format_from_template(template)


class ReasoningStreamSplitter:
    """
    Default mode = content.
    Thinking starts only after start marker is seen.
    Returns ('thinking', text) or ('content', text).

    Supports markers split across multiple chunks via small overlap buffer.
    """

    def __init__(self, fmt: Optional[ReasoningFormat]):
        self.fmt = fmt
        self.mode = "content"
        self.pending = ""

        if fmt is None:
            self.max_keep = 0
        else:
            self.max_keep = max(len(fmt.start), len(fmt.end)) - 1

    def push(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []

        if self.fmt is None:
            return [("content", text)]

        self.pending += text
        out: List[Tuple[str, str]] = []

        while True:
            if self.mode == "content":
                idx = self.pending.find(self.fmt.start)

                if idx == -1:
                    safe_len = max(0, len(self.pending) - self.max_keep)
                    if safe_len > 0:
                        piece = self.pending[:safe_len]
                        self.pending = self.pending[safe_len:]
                        if piece:
                            out.append(("content", piece))
                    break

                # content before start marker
                if idx > 0:
                    piece = self.pending[:idx]
                    if piece:
                        out.append(("content", piece))

                # consume start marker and enter thinking
                self.pending = self.pending[idx + len(self.fmt.start):]
                self.mode = "thinking"
                continue

            else:  # thinking mode
                idx = self.pending.find(self.fmt.end)

                if idx == -1:
                    safe_len = max(0, len(self.pending) - self.max_keep)
                    if safe_len > 0:
                        piece = self.pending[:safe_len]
                        self.pending = self.pending[safe_len:]
                        if piece:
                            out.append(("thinking", piece))
                    break

                # thinking before end marker
                if idx > 0:
                    piece = self.pending[:idx]
                    if piece:
                        out.append(("thinking", piece))

                # consume end marker and return to content
                self.pending = self.pending[idx + len(self.fmt.end):]
                self.mode = "content"
                continue

        return out

    def finish(self) -> List[Tuple[str, str]]:
        if not self.pending:
            return []

        piece = self.pending
        self.pending = ""

        if self.fmt is None:
            return [("content", piece)]

        return [(self.mode, piece)] if piece else []


def split_full_text_by_reasoning_format(text: str, fmt: Optional[ReasoningFormat]) -> Tuple[str, str]:
    """
    Non-stream helper. Returns (thinking, content).
    Unknown format => all content.
    """
    if not text:
        return "", ""

    splitter = ReasoningStreamSplitter(fmt)
    thinking_parts: List[str] = []
    content_parts: List[str] = []

    for kind, piece in splitter.push(text):
        if kind == "thinking":
            thinking_parts.append(piece)
        else:
            content_parts.append(piece)

    for kind, piece in splitter.finish():
        if kind == "thinking":
            thinking_parts.append(piece)
        else:
            content_parts.append(piece)

    return "".join(thinking_parts), "".join(content_parts)
