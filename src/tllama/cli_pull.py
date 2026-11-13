"""Human pull feedback over the unchanged TLlama/Ollama progress stream."""

from contextlib import aclosing
import json
import math
import os
import shutil
import sys

from tllama.cli_output import size_text
from tllama.cli_status import WaitStatus


def _text(event, *, decimal_units):
    text = event['status']
    if 'completed' in event:
        text += (f": {size_text(event['completed'], decimal_units=decimal_units)} / "
                 f"{size_text(event.get('total'), decimal_units=decimal_units)}")
    return text


def _columns():
    try:
        return os.get_terminal_size(sys.stderr.fileno()).columns
    except (OSError, AttributeError):
        return shutil.get_terminal_size().columns


def _terminal_text(event, text):
    # Leave space for the spinner and avoid wrapping into terminal history.
    width = max(0, _columns() - 3)
    completed, total = event.get('completed'), event.get('total')
    if (type(completed) in (int, float) and type(total) in (int, float)
            and math.isfinite(completed) and math.isfinite(total)
            and completed >= 0 and total > 0):
        ratio = min(completed / total, 1)
        filled = int(20 * ratio)
        bar = f" [{'#' * filled}{'-' * (20 - filled)}] {int(100 * ratio)}%"
        if len(text + bar) <= width:
            text += bar
    return text[:width]


async def pull_progress(client, model, *, as_json, decimal_units):
    previous, phase = None, None
    async with WaitStatus(_terminal_text({}, 'resolving repository'),
                          enabled=not as_json and _columns() >= 3) as status:
        async with aclosing(client.pull(model)) as progress:
            async for event in progress:
                if as_json:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    continue
                text = _text(event, decimal_units=decimal_units)
                if status.enabled:
                    current_phase = (event['status'], event.get('digest'))
                    if previous is not None and current_phase != phase:
                        status.clear()
                        print(previous, file=sys.stderr, flush=True)
                    if event['status'] == 'success':
                        status.stop()
                        print(text, file=sys.stderr, flush=True)
                    else:
                        status.label = _terminal_text(event, text)
                    phase = current_phase
                elif text != previous:
                    print(text, file=sys.stderr, flush=True)
                previous = text
