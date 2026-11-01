"""Read-only local sharing and byte accounting over a locked repository snapshot."""

import re

from tllama.repository_check import check_repository, _cancel
from tllama.repository_cleanup import _plan_group
from tllama.repository_names import definition_path
from tllama.repository_pins import snapshot
from tllama.repository_store import file_signature


def inspect_repository(store, *, model=None, group=None, stop=None):
    if model is not None and group is not None:
        raise ValueError('Select a model or a group, not both.')
    report = None
    try:
        with store.lock(create=False):
            report = check_repository(store, stop=stop)
            report['operation'] = 'inspect' if model is not None or group is not None else 'stats'
            report['selection'] = {'model': model, 'group': group}
            selected = None
            if model is not None:
                definition_path(model)  # Validate a logical name without touching the filesystem.
                selected = next((row['groups'] for row in report['definitions'] if row['name'] == model), None)
                if selected is None:
                    raise ValueError(f'Model definition not found: {model}')
            elif group is not None:
                if not re.fullmatch('[0-9a-f]{64}', group):
                    raise ValueError('Group must be a lowercase SHA-256 identifier.')
                if not any(row['id'] == group for row in report['groups']):
                    raise ValueError(f'Data group not found: {group}')
                selected = [group]
            pins = snapshot(store) if report['reservations_complete'] else None
            unique, total, counts = {}, 0, {key: 0 for key in ('referenced', 'reserved', 'orphan', 'unknown')}
            candidate_bytes = 0
            for row in report['groups']:
                _cancel(stop)
                row['weight_bytes'] = None
                row['prune_candidate'] = None
                counts[row['usage']] += 1
                if not row['valid']:
                    continue
                size = 0
                for relative in row['files']:
                    device, inode, length, _ = file_signature(store.root / relative)
                    unique[(device, inode)] = length
                    size += length
                row['weight_bytes'] = size
                total += size
                if report['ok']:
                    row['prune_candidate'] = (row['usage'] == 'orphan' and
                        _plan_group(store, row['id'], [path.rsplit('/', 1)[-1] for path in row['files']], pins) is not None)
                    if row['prune_candidate']:
                        candidate_bytes += size
            sizes = {row['id']: row['weight_bytes'] for row in report['groups']}
            for row in report['definitions']:
                values = [sizes.get(key) for key in row['groups']]
                row['weight_bytes'] = sum(values) if all(value is not None for value in values) else None
            report['summary'].update(groups_by_usage=counts,
                shared_groups=sum(len(row['references']) > 1 for row in report['groups']),
                valid_group_weight_bytes=total, valid_group_unique_inode_bytes=sum(unique.values()),
                byte_counts_complete=report['ok'],
                prune_candidate_weight_bytes=candidate_bytes if report['ok'] else None)
            report['selected_groups'] = selected if selected is not None else [row['id'] for row in report['groups']]
    except (OSError, ValueError, RuntimeError) as exc:
        if report is None:
            report = {'version': 1, 'operation': 'inspect' if model is not None or group is not None else 'stats',
                      'issues': [], 'summary': {}, 'groups': [], 'definitions': [], 'selected_groups': []}
        report['ok'], report['complete'] = False, False
        report['summary'].update(byte_counts_complete=False, prune_candidate_weight_bytes=None)
        report['issues'].append({'severity':'error', 'code':'inspection_failed', 'path':'.', 'message':str(exc)})
        report['summary']['errors'] = sum(issue['severity'] == 'error' for issue in report['issues'])
        for row in report['groups']:
            row['prune_candidate'] = None
    return report
