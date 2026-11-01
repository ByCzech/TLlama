"""Local whole-store orphan group pruning with a read-only default preview.

Each invocation makes a fresh inventory. Apply holds the same repository lock
from that inventory through durable scheduling and unlink; no saved report can
authorize a deletion. Standalone cache/staging cleanup is a separate policy.
"""

from pathlib import Path

from tllama.repository_check import check_repository, CheckCancelled, _cancel
from tllama.repository_cleanup import _read_queue, _save_queue, _plan_group, _remove_group
from tllama.repository_pins import snapshot


def _issue(report, code, path, error):
    report['issues'].append({'severity': 'error', 'code': code, 'path': path, 'message': str(error)})


def prune_repository(store, *, apply=False, stop=None):
    if type(apply) is not bool:
        raise ValueError('apply must be boolean.')
    report = {'version': 1, 'operation': 'prune', 'apply': apply,
        'ok': False, 'complete': False, 'can_apply': False, 'cancelled': False,
        'candidates': [], 'referenced_groups': [], 'reserved_groups': [],
        'deferred_groups': [], 'removed_groups': [], 'removed_files': [], 'issues': []}
    try:
        with store.lock(create=False):
            _cancel(stop)
            checked = check_repository(store, stop=stop)
            report['issues'] = [issue for issue in checked['issues'] if issue['code'] != 'unreferenced_group']
            report['complete'] = checked['complete']
            # No deletion with missing, malformed or incomplete protection/data.
            if not checked['ok']:
                return _finish(report)
            pins = snapshot(store)
            pending = _read_queue(store)
            for row in checked['groups']:
                _cancel(stop)
                group = row['id']
                if row['usage'] == 'referenced':
                    report['referenced_groups'].append(group)
                    continue
                if row['usage'] == 'reserved':
                    report['reserved_groups'].append(group)
                    continue
                if row['usage'] != 'orphan' or not row['valid']:
                    _issue(report, 'group_not_proved_unused', row['path'], 'Cannot prove safe group ownership.')
                    continue
                files = [Path(path).name for path in row['files']]
                try:
                    plan = _plan_group(store, group, files, pins)
                    if plan is None:
                        report['deferred_groups'].append(group)
                        continue
                    phases, _ = plan
                    paths = [path.relative_to(store.root).as_posix() for phase in phases for path in phase
                             if path.exists()]
                    if group in pending and pending[group] != files:
                        raise ValueError('Pending cleanup disagrees with the current complete group.')
                    report['candidates'].append({'group': group, 'files': files, 'remove_paths': paths})
                except (OSError, ValueError) as exc:
                    _issue(report, 'group_plan_failed', row['path'], exc)
            report['can_apply'] = not any(issue['severity'] == 'error' for issue in report['issues'])
            _cancel(stop)
            if not apply or not report['can_apply'] or not report['candidates']:
                return _finish(report)
            # Schedule the complete set before the first destructive operation.
            # A crash/failed unlink remains recoverable by the ordinary collector.
            for row in report['candidates']:
                pending[row['group']] = row['files']
            _save_queue(store, pending)
            for row in report['candidates']:
                _cancel(stop)
                group = row['group']
                # _remove_group repeats path preflight without releasing this
                # lock. Finish one group before honoring a terminal stop event.
                if not _remove_group(store, group, row['files'], pins, report['removed_files']):
                    report['deferred_groups'].append(group)
                    continue
                report['removed_groups'].append(group)
                del pending[group]
                _save_queue(store, pending)
            _cancel(stop)
    except CheckCancelled:
        report['cancelled'], report['complete'] = True, False
        report['can_apply'] = False
    except (OSError, ValueError, RuntimeError) as exc:
        _issue(report, 'prune_failed', '.', exc)
        report['can_apply'] = False
    return _finish(report)


def _finish(report):
    errors = sum(issue['severity'] == 'error' for issue in report['issues'])
    report['ok'] = report['complete'] and not report['cancelled'] and not errors
    report['summary'] = {
        'candidate_groups': len(report['candidates']),
        'candidate_paths': sum(len(row['remove_paths']) for row in report['candidates']),
        'referenced_groups': len(report['referenced_groups']),
        'reserved_groups': len(report['reserved_groups']),
        'deferred_groups': len(report['deferred_groups']),
        'removed_groups': len(report['removed_groups']), 'removed_paths': len(report['removed_files']),
        'errors': errors, 'warnings': sum(issue['severity'] == 'warning' for issue in report['issues'])}
    return report
