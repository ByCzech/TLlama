"""Human formatting follows locale and client timezone without changing JSON."""

from datetime import datetime, timezone
import json
import locale
import os
import time

import pytest

from tllama.cli import main
from tllama.cli_locale import configure_locale, number_text, timestamp_text
from tllama.cli_output import processes_text, models_text
from tllama.cli_metrics import statistics
from test_client_cli import cli_server, isolated_proxy_environment


@pytest.fixture(autouse=True)
def restore_locale_and_timezone():
    categories = {category: locale.setlocale(category) for category in (locale.LC_NUMERIC, locale.LC_TIME)}
    old_tz = os.environ.get('TZ')
    locale.setlocale(locale.LC_TIME, 'C')
    try:
        yield
    finally:
        for category, value in categories.items():
            locale.setlocale(category, value)
        if old_tz is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = old_tz
        time.tzset()


@pytest.fixture
def czech_numbers(monkeypatch):
    convention = dict(locale.localeconv(), decimal_point=',', thousands_sep='\u202f', grouping=[3, 0])
    monkeypatch.setattr(locale, 'localeconv', lambda: convention)


def patterns(monkeypatch, date, clock, *, tz='UTC'):
    monkeypatch.setenv('TZ', tz)
    time.tzset()
    formats = {'D_FMT': date, 'T_FMT': clock, 'T_FMT_AMPM': '%I:%M:%S %p'}
    monkeypatch.setattr('tllama.cli_locale._locale_format', lambda name, fallback: formats.get(name, fallback))


@pytest.mark.parametrize('date,clock,expected', [
    ('%-d.%-m.%Y', '%H:%M:%S', '23.8. 15:47'),
    ('%d.%m.%Y', '%T', '23.08. 15:47'),
    ('%m/%d/%Y', '%r', '08/23 03:47 PM'),
    ('%Y-%m-%d', '%H:%M:%S %Z', '08-23 15:47'),
    ('%Y年%m月%d日', '%H時%M分%S秒', '08月23日 15時47分'),
    ('%d de %B de %Y', '%R', '23 de August 15:47'),
    ('%D', '%T', '08/23 15:47'),
    ('%F', '%T', '08-23 15:47'),
])
def test_yearless_date_and_minute_clock_keep_locale_order(monkeypatch, date, clock, expected):
    patterns(monkeypatch, date, clock)
    assert timestamp_text('2026-08-23T15:47:59.123456Z',
                          now=datetime(2026, 6, 1, tzinfo=timezone.utc)) == expected


@pytest.mark.parametrize('value,expected', [
    ('2026-10-22T22:15:12.482903+00:00', '00:15'),
    ('2026-11-02T10:20:59.999999Z', '2.11. 11:20'),
    ('2027-01-03T09:01:34.123456Z', '3.1.2027 10:01'),
    ('2026-03-29T00:59:59Z', '29.3. 01:59'),
    ('2026-03-29T01:00:00Z', '29.3. 03:00'),
])
def test_prague_local_day_year_and_dst_boundaries(monkeypatch, value, expected):
    patterns(monkeypatch, '%-d.%-m.%Y', '%H:%M:%S', tz='Europe/Prague')
    assert timestamp_text(value, now=datetime(2026, 10, 23, 10, tzinfo=timezone.utc)) == expected


@pytest.mark.parametrize('bad', [None, 123, '', 'nonsense', '2026-10-22T22:15:12', '99999-01-01T00:00:00Z'])
def test_unknown_timestamps_do_not_leak_raw_data_or_guess_timezone(bad):
    assert timestamp_text(bad) == 'unknown'


def test_numeric_grouping_applies_to_tables_and_all_verbose_measurements(czech_numbers):
    assert number_text(262144) == '262\u202f144'
    model = {'name': 'm', 'size': 16773098312, 'size_vram': 16746961480,
             'context_length': 262144, 'expires_at': None}
    text = processes_text({'models': [model]})
    assert '16,8 GB' in text and '0,2% CPU / 99,8% GPU' in text
    assert '262\u202f144' in text and 'pinned' in text
    stats = statistics({'eval_count': 12345, 'eval_duration': 1_000_000_000}, elapsed=2.5, first_text=0.25)
    assert '2,500 s' in stats and '0,250 s' in stats
    assert '12\u202f345 tokens' in stats and '12\u202f345,00 tok/s' in stats


def test_list_modification_time_uses_the_same_short_local_format(monkeypatch):
    patterns(monkeypatch, '%d.%m.%Y', '%H:%M:%S')
    # Future year avoids depending on today's date for the presentation test.
    text = models_text({'models': [{'name': 'm', 'modified_at': '2099-01-02T03:04:05Z'}]})
    assert '02.01.2099 03:04' in text and 'T03:04:05' not in text


def test_json_keeps_original_precision_offsets_and_numeric_types(cli_server, capsys, czech_numbers):
    payload = {'models': [{'name': 'm', 'size': 16773098312, 'size_vram': 16746961480,
                          'context_length': 262144, 'expires_at': '2026-10-22T22:15:12.482903+00:00'}]}
    cli_server['responses']['/api/ps'] = (200, payload)
    assert main(['ps', '--host', cli_server['url'], '--json']) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_real_client_ps_localizes_the_captured_memory_and_timestamp(cli_server, capsys, czech_numbers, monkeypatch):
    patterns(monkeypatch, '%d.%m.%Y', '%H:%M:%S', tz='Europe/Prague')
    cli_server['responses']['/api/ps'] = (200, {'models': [{
        'name': 'm', 'size': 16773098312, 'size_vram': 16746961480,
        'context_length': 262144, 'expires_at': '2099-10-22T22:15:12.482903+00:00'}]})
    assert main(['ps', '--host', cli_server['url']]) == 0
    out = capsys.readouterr().out
    assert '0,2% CPU / 99,8% GPU' in out and '262\u202f144' in out
    assert '23.10.2099 00:15' in out and '482903' not in out and '+00:00' not in out


def test_invalid_os_locale_falls_back_without_breaking_client(monkeypatch):
    monkeypatch.setenv('LC_ALL', 'not_an_installed_locale')
    configure_locale()
    assert locale.setlocale(locale.LC_TIME) == 'C'
    assert locale.setlocale(locale.LC_NUMERIC) == 'C'
