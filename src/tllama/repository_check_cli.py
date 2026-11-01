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


def prune_text(data):
    mode = 'APPLY' if data['apply'] else 'PREVIEW'
    status = 'CANCELLED' if data['cancelled'] else 'OK' if data['ok'] else 'FAILED'
    counts = data['summary']
    lines = [f'Repository prune: {mode} {status}',
        f"Candidate groups: {counts['candidate_groups']}; removed: {counts['removed_groups']}; "
        f"referenced: {counts['referenced_groups']}; reserved: {counts['reserved_groups']}; "
        f"deferred: {counts['deferred_groups']}"]
    for row in data['candidates']:
        lines.append(f"group: data/{row['group']}/")
        lines.extend('  ' + path for path in row['remove_paths'])
    if data['apply']:
        lines.extend('removed: ' + path for path in data['removed_files'])
    elif data['can_apply']:
        lines.append('Preview only. Use --apply to recheck and remove currently unused groups.')
    lines.extend(f"{issue['severity']}: {issue['path']}: {issue['code']}: {issue['message']}"
                 for issue in data['issues'])
    return '\n'.join(lines)


def run_prune(root, *, apply=False, as_json=False):
    from tllama.repository_prune import prune_repository
    stop = threading.Event()
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, lambda *args: stop.set())
    try:
        report = prune_repository(RepositoryStore(root), apply=apply, stop=stop)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False) if as_json else prune_text(report))
        if report['cancelled'] or stop.is_set():
            return 130
        return 0 if report['ok'] else 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def inspection_text(data):
    counts = data['summary']
    lines = [f"Repository {data['operation']}: {'OK' if data['ok'] else 'FAILED'}"]
    if 'valid_group_weight_bytes' in counts:
        lines.append(f"Valid group weight bytes: {counts['valid_group_weight_bytes']}; "
                     f"unique inode bytes: {counts['valid_group_unique_inode_bytes']}")
        lines.append(f"Groups by usage: {counts['groups_by_usage']}; shared groups: {counts['shared_groups']}")
        lines.append(f"Prune candidate weight bytes: {counts['prune_candidate_weight_bytes']} "
                     '(snapshot; not disk space guaranteed to be freed)')
    for row in data['groups']:
        if data['operation'] != 'inspect' or row['id'] not in data.get('selected_groups', []):
            continue
        lines.append(f"{row['path']}/: {row['usage']}; reserved={row['reserved']}; "
                     f"bytes={row.get('weight_bytes')}; prune_candidate={row.get('prune_candidate')}")
        lines.append('  definitions: ' + (', '.join(row['references']) or '(none)'))
        lines.extend('  ' + path for path in row['files'])
    lines.extend(f"{issue['severity']}: {issue['path']}: {issue['message']}" for issue in data['issues'])
    return '\n'.join(lines)


def run_inspection(root, *, model=None, group=None, as_json=False):
    from tllama.repository_inspect import inspect_repository
    stop = threading.Event()
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, lambda *args: stop.set())
    try:
        report = inspect_repository(RepositoryStore(root), model=model, group=group, stop=stop)
        if stop.is_set():
            return 130
        print(json.dumps(report, ensure_ascii=False, allow_nan=False) if as_json else inspection_text(report))
        return 0 if report['ok'] else 1
    except CheckCancelled:
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
