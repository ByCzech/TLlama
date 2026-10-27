"""Shared warn/strict policy, feature audit boundaries and real worker startup."""

from dataclasses import replace
import logging
from pathlib import Path

import pytest

from tllama.decision_contract import DECISION_RUNTIME
from tllama.runtime_fingerprint import evaluate_runtime_fingerprint, RuntimeFingerprintMismatch
from tllama.worker import WorkerStartupError
from tllama.worker_config import WorkerLaunchSpec, RUNTIME_FINGERPRINT, SUPPORTED_RUNTIME_FINGERPRINTS
from test_native_worker import factory, records


BUILD = 'b11850-0e9433bfd3'
MODES = ['generation', 'embedding', 'laya', 'nimble', 'clef']


def launch(mode, **kwargs):
    return WorkerLaunchSpec(model=Path('/models/a.gguf'), device='none',
        embedding=mode == 'embedding', decision_type=mode if mode in MODES[2:] else None, **kwargs)


@pytest.mark.parametrize('policy', ['warn', 'strict'])
def test_matching_runtime_is_silent(policy, caplog):
    assert evaluate_runtime_fingerprint(BUILD, expected_fingerprints={BUILD}, policy=policy)
    assert not caplog.records


@pytest.mark.parametrize('observed', [None, '', 'unknown-build'])
def test_unknown_runtime_warns_or_refuses_with_the_same_identity_evidence(observed, caplog):
    with caplog.at_level(logging.WARNING):
        assert not evaluate_runtime_fingerprint(observed, expected_fingerprints={BUILD},
                                                policy='warn', context='test operation')
    assert len(caplog.records) == 1
    assert 'Unverified native runtime fingerprint' in caplog.text and BUILD in caplog.text
    assert 'test operation' in caplog.text
    with pytest.raises(RuntimeFingerprintMismatch, match='Unsupported runtime fingerprint'):
        evaluate_runtime_fingerprint(observed, expected_fingerprints={BUILD}, policy='strict')


@pytest.mark.parametrize('expected,policy', [([], 'warn'), ([None], 'warn'), ([' '], 'strict'),
    (['a\0b'], 'strict'), ([' a '], 'warn'), ('build', 'warn'), (['build'], 'ignore')])
def test_invalid_policy_inputs_fail_without_warning(expected, policy, caplog):
    with pytest.raises(ValueError):
        evaluate_runtime_fingerprint('build', expected_fingerprints=expected, policy=policy)
    assert not caplog.records


@pytest.mark.parametrize('mode', MODES)
def test_explicit_pin_overrides_only_this_launch_in_every_mode(mode):
    original = launch(mode)
    expected = frozenset({DECISION_RUNTIME}) if mode in MODES[2:] else SUPPORTED_RUNTIME_FINGERPRINTS
    assert original.accepted_runtime_fingerprints == expected
    # Also exercise pinning to the default build: no magic-default sentinel.
    for fingerprint in (BUILD, RUNTIME_FINGERPRINT):
        pinned = replace(original, expected_runtime_fingerprint=fingerprint, version_policy='strict')
        assert pinned.accepted_runtime_fingerprints == {fingerprint}
        assert original.accepted_runtime_fingerprints == expected
    assert BUILD not in SUPPORTED_RUNTIME_FINGERPRINTS and DECISION_RUNTIME != BUILD


@pytest.mark.parametrize('value', ['', ' ', ' build', 'build ', 'a\0b', 1, False])
def test_invalid_explicit_pin_is_rejected(value):
    with pytest.raises(ValueError, match='expected_runtime_fingerprint'):
        launch('generation', expected_runtime_fingerprint=value)


@pytest.mark.parametrize('factory', ['tcp'], indirect=True)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('policy', ['warn', 'strict'])
async def test_real_worker_unknown_build_respects_the_same_policy(factory, mode, policy, caplog):
    env = {'STUB_BUILD': BUILD}
    if mode in MODES[2:]:
        env.update(STUB_DECISION='normal', STUB_DECISION_KIND=mode)
    worker, root = factory(embedding=mode == 'embedding',
        decision_type=mode if mode in MODES[2:] else None,
        version_policy=policy, environment=env, memory_source='log')
    if policy == 'strict':
        with pytest.raises(WorkerStartupError, match='Unsupported runtime fingerprint'):
            await worker.start()
        assert not worker.runtime_directory.exists() and worker.process.returncode is not None
        assert not any(r['path'] in {'/v1/systemone', '/v1/chat/completions', '/embeddings'}
                       for r in records(root, 'requests'))
    else:
        with caplog.at_level(logging.WARNING):
            async with worker:
                assert worker.runtime_fingerprint == BUILD
                if mode in MODES[2:]:
                    result = await worker.decision({'state': 'charged twice', 'questions': {
                        'q': {'type': 'noul', 'instructions': 'Is the customer angry?'}}})
                    assert result['answers']
                else:
                    assert await worker.tokenize('same worker stays usable')
        assert 'Unverified native runtime fingerprint' in caplog.text


@pytest.mark.parametrize('factory', ['tcp'], indirect=True)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('observed,accepted', [(BUILD, True), (DECISION_RUNTIME, False), (None, False)])
async def test_real_worker_strict_exact_pin_precedes_all_feature_requests(factory, mode, observed, accepted):
    env = {'STUB_BUILD': observed or BUILD}
    if mode in MODES[2:]:
        env.update(STUB_DECISION='normal', STUB_DECISION_KIND=mode)
    worker, root = factory('missing-build' if observed is None else 'normal', environment=env,
        embedding=mode == 'embedding', decision_type=mode if mode in MODES[2:] else None,
        expected_runtime_fingerprint=BUILD, version_policy='strict', memory_source='log')
    if accepted:
        async with worker:
            assert worker.runtime_fingerprint == BUILD
    else:
        with pytest.raises(WorkerStartupError, match='Unsupported runtime fingerprint'):
            await worker.start()
        assert worker.process.returncode is not None and not worker.runtime_directory.exists()
        assert not any(r['path'] in {'/v1/systemone', '/v1/chat/completions', '/embeddings'}
                       for r in records(root, 'requests'))
