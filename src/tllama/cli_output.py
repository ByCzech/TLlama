"""Human-readable views of the server's public responses."""

from __future__ import annotations

import math

from tllama.client import ClientError
from tllama.cli_locale import number_text, timestamp_text


def size_text(value):
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        return 'unknown'
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1000 or unit == 'TB':
            return f'{number_text(value, 0)} B' if unit == 'B' else f'{number_text(value, 1)} {unit}'
        value /= 1000


def _id(model):
    return str(model.get('digest') or '').removeprefix('sha256:')[:12] or '-'


def _table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in rows)]) for i, header in enumerate(headers)]
    return '\n'.join('  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
                     for row in [headers, *rows])


def models_text(data):
    return _table(['NAME', 'ID', 'SIZE', 'MODIFIED'], [
        [m.get('name') or m.get('model', '-'), _id(m), size_text(m.get('size')),
         timestamp_text(m['modified_at']) if m.get('modified_at') else '-']
        for m in data['models']])


def _processor(model):
    total, device = model.get('size'), model.get('size_vram')
    if (type(total) not in {int, float} or type(device) not in {int, float}
            or not math.isfinite(total) or not math.isfinite(device) or total <= 0 or not 0 <= device <= total):
        return 'unknown'
    if device == total:
        return '100% GPU'
    if device == 0:
        return '100% CPU'
    gpu = 100 * device / total
    return f'{number_text(100 - gpu, 1)}% CPU / {number_text(gpu, 1)}% GPU'


def processes_text(data):
    return _table(['NAME', 'ID', 'SIZE', 'PROCESSOR', 'CONTEXT', 'RUNNER', 'EXPIRES'], [
        [m.get('name') or m.get('model', '-'), _id(m), size_text(m.get('size')), _processor(m),
         number_text(m['context_length']) if type(m.get('context_length')) is int else 'unknown',
         m.get('runner') or 'unknown', timestamp_text(m['expires_at']) if m.get('expires_at') else 'pinned']
        for m in data['models']])


def show_text(model, data, field=None):
    if field:
        value = data.get(field, '')
        if not isinstance(value, str):
            raise ClientError(f'Server returned a non-text {field} field.')
        return value
    details = data.get('details') or {}
    if not isinstance(details, dict):
        raise ClientError('Server returned invalid model details.')
    caps = data.get('capabilities') or []
    if not isinstance(caps, list) or any(not isinstance(c, str) for c in caps):
        raise ClientError('Server returned invalid model capabilities.')
    lines = [f'Model: {model}', f'Architecture: {details.get("family", "unknown")}',
             f'Parameters: {details.get("parameter_size", "unknown")}',
             f'Quantization: {details.get("quantization_level", "unknown")}',
             f'Capabilities: {", ".join(caps) or "unknown"}']
    if data.get('system'):
        lines += ['', 'System:', str(data['system'])]
    if data.get('parameters'):
        lines += ['', 'Parameters:', str(data['parameters'])]
    return '\n'.join(lines)
