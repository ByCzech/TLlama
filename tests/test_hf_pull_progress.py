"""HTTP/Xet progress must count file bytes once, without rendering a bar."""

import logging
import sys
from types import SimpleNamespace

import pytest

from tllama.backend import PullProgress, _hf_tqdm_class_for


def test_http_resume_and_retry_correction():
    progress = PullProgress()
    with _hf_tqdm_class_for(progress)(total=100, initial=30) as bar:
        bar.update(20)
        bar.update_transfer(20)
        assert progress.snapshot() == (50, 100)
        # A server ignoring Range restarts the download from zero.
        bar.update(-50)
        bar.update_transfer(-50)
        bar.update(100)
        bar.update_transfer(100)
        assert bar.n == 100
    assert progress.snapshot() == (100, 100)


def test_total_discovered_after_construction():
    progress = PullProgress(total=100)
    bar = _hf_tqdm_class_for(progress)(total=None)
    assert progress.snapshot() == (0, 100)
    bar.total = 120
    bar.update(40)
    assert progress.snapshot() == (40, 120)


def test_failed_download_does_not_report_completion():
    progress = PullProgress()
    with pytest.raises(OSError, match="connection lost"):
        with _hf_tqdm_class_for(progress)(total=100) as bar:
            bar.update(40)
            raise OSError("connection lost")
    assert progress.snapshot() == (40, 100)


@pytest.mark.parametrize("with_progress", [False, True])
def test_download_signature_and_progress(manager, monkeypatch, with_progress):
    # Explicit modern signature rejects deprecated local_dir_use_symlinks.
    def download(*, repo_id, filename, revision, token, local_dir, tqdm_class):
        assert (repo_id, filename, revision, token) == (
            "owner/model", "weights.gguf", "revision", "token"
        )
        assert local_dir == manager.hf_models_dir / "owner" / "model"
        assert local_dir.is_dir()
        if with_progress:
            with tqdm_class(total=100) as bar:
                bar.update(100)
                bar.set_postfix_str("speed", refresh=False)
        else:
            assert tqdm_class is None
        return str(local_dir / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    progress = PullProgress() if with_progress else None
    result = manager._pull_hf_file_sync(
        "owner/model", "weights.gguf", token="token", revision="revision", progress=progress
    )
    assert result == str(manager.hf_models_dir / "owner/model/weights.gguf")
    if progress is not None:
        assert progress.snapshot() == (100, 100)


@pytest.mark.parametrize("network_bytes", [60, 150])
@pytest.mark.parametrize("external_bar", [False, True])
def test_real_xet_reporter_counts_reconstructed_bytes_only(network_bytes, external_bar, capsys):
    # Optional dependency: run this against installed supported HF releases.
    module = pytest.importorskip("huggingface_hub.utils._xet_progress_reporting")
    progress = PullProgress()
    cls = _hf_tqdm_class_for(progress)
    kwargs = {"external_reconstruction_bar": cls(total=None)} if external_bar else {"tqdm_class": cls}
    with module.XetDownloadProgressReporter(
        reconstruction_desc="weights.gguf", total=None, log_level=logging.INFO, **kwargs
    ) as reporter:
        assert reporter.transfer_bar is reporter.reconstruction_bar
        for completed, transferred in [(0, network_bytes // 2), (40, network_bytes), (100, network_bytes)]:
            reporter.update_progress(SimpleNamespace(
                total_bytes_completed=completed,
                total_transfer_bytes_completed=transferred,
                total_bytes_completion_rate=10,
                total_transfer_bytes_completion_rate=20,
                total_bytes=100,
            ))
            assert progress.snapshot() == (completed, 100)
    assert progress.snapshot() == (100, 100)
    assert capsys.readouterr() == ("", "")
