"""A second --force must not take the backup the first one made.

The .toml.bak exists because --force is destructive and the person
reaching for it is thinking about what they want to gain, not what they
are about to lose. Running it twice used to overwrite that copy with a
copy of the generated file -- so the second run destroyed the only thing
protecting against the first, and the further along the mistake went the
less there was left to recover.
"""

import pytest


GGUF = b"gguf-bytes"

EDITED = '[llm]\nmodel = "Local/model.gguf"\n\n[runtime]\nn_ctx = 8192\n'


@pytest.fixture
def repo_with_a_backup(manager, gguf_file, monkeypatch):
    """A model whose .toml has already been replaced once."""
    gguf_file("Local/model.gguf", GGUF)

    monkeypatch.setattr(
        manager, "_get_model_metadata_sync", lambda path: {"name": "model"}
    )

    toml_path = manager.local_models_dir / "model.toml"
    toml_path.write_text(EDITED, encoding="utf-8")

    manager.rebuild_repository(force=True)

    backup = toml_path.with_name(toml_path.name + ".bak")
    assert backup.exists(), "the first --force should have made a backup"
    assert backup.read_text(encoding="utf-8") == EDITED

    return manager, toml_path, backup


class TestASecondForce:
    def test_the_file_is_skipped_rather_than_overwritten(self, repo_with_a_backup):
        manager, toml_path, backup = repo_with_a_backup

        results = manager.rebuild_repository(force=True)

        assert [r["action"] for r in results] == ["skip"]
        assert "backup" in results[0]["reason"]

    def test_the_first_backup_survives(self, repo_with_a_backup):
        """The point of the refusal: what the person wrote is still there."""
        manager, toml_path, backup = repo_with_a_backup

        manager.rebuild_repository(force=True)

        assert backup.read_text(encoding="utf-8") == EDITED

    def test_the_dry_run_reports_the_skip_and_not_an_overwrite(
        self, repo_with_a_backup
    ):
        """A dry run that promises what the real run refuses is worse than none."""
        manager, toml_path, backup = repo_with_a_backup

        results = manager.rebuild_repository(force=True, dry_run=True)

        assert [r["action"] for r in results] == ["skip"]

    def test_one_file_with_a_backup_does_not_stop_the_others(
        self, manager, gguf_file, monkeypatch, repo_with_a_backup
    ):
        """The refusal is per file, not an abort of the sweep."""
        manager, toml_path, backup = repo_with_a_backup
        gguf_file("Local/second.gguf", GGUF)

        results = manager.rebuild_repository(force=True)

        by_action = {}
        for entry in results:
            by_action.setdefault(entry["action"], []).append(entry["toml"].name)

        assert by_action["skip"] == ["model.toml"]
        assert by_action["create"] == ["second.toml"]


class TestOverwriteBackups:
    def test_it_replaces_the_backup_when_asked(self, repo_with_a_backup):
        manager, toml_path, backup = repo_with_a_backup

        results = manager.rebuild_repository(force=True, overwrite_backups=True)

        assert [r["action"] for r in results] == ["overwrite"]
        assert backup.read_text(encoding="utf-8") != EDITED

    def test_it_does_nothing_without_force(self, repo_with_a_backup):
        """On its own it is not a licence to replace the .toml itself."""
        manager, toml_path, backup = repo_with_a_backup

        results = manager.rebuild_repository(overwrite_backups=True)

        assert [r["action"] for r in results] == ["skip"]
        assert backup.read_text(encoding="utf-8") == EDITED


class TestTheFirstForceIsUnchanged:
    def test_a_model_with_no_backup_yet_is_still_replaced(
        self, manager, gguf_file, monkeypatch
    ):
        gguf_file("Local/model.gguf", GGUF)
        monkeypatch.setattr(
            manager, "_get_model_metadata_sync", lambda path: {"name": "model"}
        )
        toml_path = manager.local_models_dir / "model.toml"
        toml_path.write_text(EDITED, encoding="utf-8")

        results = manager.rebuild_repository(force=True)

        assert [r["action"] for r in results] == ["overwrite"]
        assert toml_path.with_name(toml_path.name + ".bak").read_text(
            encoding="utf-8"
        ) == EDITED
