"""Local daemon maintenance commands; never start or contact an HTTP server."""

import json
import signal
import threading

from tllama.repository_check import CheckCancelled, check_repository
from tllama.repository_store import RepositoryStore


def report_text(data):
    counts = data['summary']
    status = 'OK' if data['ok'] else 'FAILED'
    mode = 'full content hashes' if data['hashes'] else 'structure and headers; content hashes not checked'
    lines = [f'Repository check: {status} ({mode})',
             f"Definitions: {counts['definitions']}; groups: {counts['groups']}; caches: {counts['caches']}; "
             f"errors: {counts['errors']}; warnings: {counts['warnings']}"]
    lines.extend(f"{issue['severity']}: {issue['path']}: {issue['code']}: {issue['message']}"
                 for issue in data['issues'])
    return '\n'.join(lines)


def run_check(root, *, hashes=False, as_json=False):
    stop = threading.Event()
    previous = None
    # Turn terminal Ctrl+C into cooperative cancellation. It cannot interrupt
    # reservation acquisition/closure half way through their filesystem work.
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, lambda *args: stop.set())
    try:
        report = check_repository(RepositoryStore(root), hashes=hashes, stop=stop)
        if stop.is_set():
            return 130
        print(json.dumps(report, ensure_ascii=False, allow_nan=False) if as_json else report_text(report))
        return 0 if report['ok'] else 1
    except CheckCancelled:
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
