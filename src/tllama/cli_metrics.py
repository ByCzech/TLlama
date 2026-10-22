"""Display measured client/server timings without inventing missing statistics."""

import math
from tllama.cli_locale import number_text


def _seconds(value):
    if type(value) not in {int, float} or value < 0:
        return None
    try:
        result = value / 1_000_000_000
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def statistics(data, *, elapsed, first_text=None):
    rows = [('Client total', f'{number_text(elapsed, 3)} s')]
    if first_text is not None:
        rows.append(('First text', f'{number_text(first_text, 3)} s'))
    for key, label in (('total_duration', 'Server total'), ('load_duration', 'Server preparation')):
        duration = _seconds(data.get(key))
        if duration is not None:
            rows.append((label, f'{number_text(duration, 3)} s'))
    for key, label in (('prompt_eval_count', 'Prompt evaluated'),
                       ('prompt_eval_cached_count', 'Prompt cached'), ('eval_count', 'Generated')):
        count = _count(data.get(key))
        if count is not None:
            rows.append((label, f'{number_text(count)} tokens'))
    usage = data.get('usage')
    if isinstance(usage, dict):
        for key, label in (('input_tokens', 'Input'), ('output_tokens', 'Output')):
            count = _count(usage.get(key))
            if count is not None:
                rows.append((label, f'{number_text(count)} tokens'))
    for key, count_key, label in (('prompt_eval_duration', 'prompt_eval_count', 'Prompt evaluation'),
                                  ('eval_duration', 'eval_count', 'Generation')):
        duration = _seconds(data.get(key))
        if duration is not None:
            text = f'{number_text(duration, 3)} s'
            count = _count(data.get(count_key))
            if duration > 0 and count is not None:
                try:
                    rate = count / duration
                    if math.isfinite(rate):
                        text += f' ({number_text(rate, 2)} tok/s)'
                except OverflowError:
                    pass
            rows.append((label, text))
    reason = data.get('done_reason')
    if isinstance(reason, str) and reason:
        rows.append(('Stop reason', reason))
    return 'Statistics:\n' + '\n'.join(f'  {label + ":":<22} {value}' for label, value in rows)
