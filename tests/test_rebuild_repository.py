"""`tllama rebuildrepo`: giving every .gguf on disk a definition.

Not only a one-off migration for a repository that predates .toml. A file
copied into Local/ by hand is invisible for exactly the same reason, so
this is safe and useful to run at any time -- which is what makes the
default behaviour matter: without --force it only ever adds.
"""

import gguf
import pytest

from tllama.cli import build_parser, main as cli_main
from tllama.helpers.model_toml import parse_model_toml


@pytest.fixture
def place_gguf(manager):
    def factory(relative_path, architecture="llama", **keys):
        path = manager.models_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = gguf.GGUFWriter(path=str(path), arch=architecture)
        for key, value in keys.items():
            key = key.replace("__", ".")
            if isinstance(value, bool):
                writer.add_bool(key, value)
            elif isinstance(value, int):
                writer.add_uint32(key, value)
            else:
                writer.add_string(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        return path

    return factory


def actions(results):
    return {r["action"] for r in results}


def by_action(results, action):
    return [r for r in results if r["action"] == action]


class TestTheDefaultRun:
    def test_a_file_with_no_definition_gets_one(self, manager, place_gguf):
        place_gguf("Local/hand-copied.gguf")

        results = manager.rebuild_repository()

        assert by_action(results, "create")[0]["toml"].name == "hand-copied.toml"

    def test_the_model_is_then_listed(self, manager, place_gguf):
        """The point of the whole exercise, stated as the user sees it."""
        place_gguf("Local/hand-copied.gguf")

        manager.rebuild_repository()

        assert [m["id"] for m in manager._list_local_models_sync()] == ["hand-copied"]

    def test_an_existing_definition_is_left_alone(self, manager, place_gguf):
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()
        (manager.local_models_dir / "model.toml").write_text(
            '# my notes\n[llm]\nmodel = "Local/model.gguf"\n[runtime]\nn_ctx = 8192\n',
            encoding="utf-8",
        )

        results = manager.rebuild_repository()

        assert by_action(results, "create") == []
        assert "my notes" in (manager.local_models_dir / "model.toml").read_text()

    def test_running_it_twice_changes_nothing_the_second_time(self, manager, place_gguf):
        place_gguf("HuggingFace/ns/repo/model.gguf")
        manager.rebuild_repository()

        results = manager.rebuild_repository()

        assert by_action(results, "create") == []
        assert by_action(results, "overwrite") == []

    def test_every_repository_is_covered(self, manager, place_gguf):
        place_gguf("Local/a.gguf")
        place_gguf("TLlama/ByCzech/b.gguf")
        place_gguf("HuggingFace/ns/repo/c.gguf")

        results = manager.rebuild_repository()

        assert len(by_action(results, "create")) == 3

    def test_a_deeply_nested_file_is_found(self, manager, place_gguf):
        """The .gguf may sit anywhere; only the .toml's depth is fixed. A
        scan that enforced the old naming depths would miss this one."""
        place_gguf("HuggingFace/ns/repo/Q4_K_M/sub/model.gguf")

        results = manager.rebuild_repository()

        assert by_action(results, "create")[0]["toml"].name == "model.toml"


class TestWhatItRefusesToTouch:
    def test_a_projector_is_skipped_with_a_reason(self, manager, place_gguf):
        place_gguf("HuggingFace/ns/repo/mmproj.gguf", architecture="clip")

        results = manager.rebuild_repository()

        assert "projector" in by_action(results, "skip")[0]["reason"]

    def test_a_continuation_shard_is_skipped(self, manager, place_gguf):
        place_gguf("Local/first.gguf", split__no=0, split__count=2)
        place_gguf("Local/second.gguf", split__no=1, split__count=2)

        results = manager.rebuild_repository()

        assert len(by_action(results, "create")) == 1
        assert "shard 1 of 2" in by_action(results, "skip")[0]["reason"]

    def test_an_unreadable_header_does_not_stop_the_rest(self, manager, place_gguf):
        """One bad file in a repository of forty is not a reason to leave
        the other thirty-nine undefined."""
        place_gguf("Local/good.gguf")
        (manager.local_models_dir / "bad.gguf").write_bytes(b"not a gguf at all")

        results = manager.rebuild_repository()

        assert len(by_action(results, "create")) == 1
        assert len(by_action(results, "skip")) == 1


class TestDryRun:
    def test_nothing_is_written(self, manager, place_gguf):
        place_gguf("Local/model.gguf")

        manager.rebuild_repository(dry_run=True)

        assert list(manager.local_models_dir.glob("*.toml")) == []

    def test_it_still_reports_what_would_happen(self, manager, place_gguf):
        place_gguf("Local/model.gguf")

        results = manager.rebuild_repository(dry_run=True)

        assert by_action(results, "create")[0]["toml"].name == "model.toml"

    def test_without_force_it_reports_no_overwrites(self, manager, place_gguf):
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()

        results = manager.rebuild_repository(dry_run=True)

        assert by_action(results, "overwrite") == []

    def test_with_force_the_overwrites_are_visible_in_advance(self, manager, place_gguf):
        """The reason to combine the two: seeing exactly which files are
        about to be replaced, while they still exist."""
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()

        results = manager.rebuild_repository(dry_run=True, force=True)

        assert by_action(results, "overwrite")[0]["toml"].name == "model.toml"
        assert (manager.local_models_dir / "model.toml").read_text().count("[llm]") == 1


class TestForce:
    def test_an_existing_definition_is_replaced(self, manager, place_gguf):
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()
        toml = manager.local_models_dir / "model.toml"
        toml.write_text('[llm]\nmodel = "Local/model.gguf"\n', encoding="utf-8")

        manager.rebuild_repository(force=True)

        assert "TLlama virtual model" in toml.read_text(encoding="utf-8")

    def test_the_replacement_still_names_the_same_file(self, manager, place_gguf):
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()

        manager.rebuild_repository(force=True)

        spec = parse_model_toml(
            (manager.local_models_dir / "model.toml").read_text(encoding="utf-8")
        )
        assert spec.llm_model == "Local/model.gguf"

    def test_no_duplicate_definition_is_created(self, manager, place_gguf):
        """Replacing must replace, not add a model_01 alongside."""
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()

        manager.rebuild_repository(force=True)

        assert len(list(manager.local_models_dir.glob("*.toml"))) == 1

    def test_what_was_replaced_is_kept_alongside(self, manager, place_gguf):
        """Unconditional, not a flag: anyone reaching for --force is
        thinking about what they want to gain, and would only think to ask
        for a copy once it was too late to make one."""
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()
        toml = manager.local_models_dir / "model.toml"
        toml.write_text(
            '# irreplaceable\n[llm]\nmodel = "Local/model.gguf"\n', encoding="utf-8"
        )

        manager.rebuild_repository(force=True)

        assert "irreplaceable" in (
            manager.local_models_dir / "model.toml.bak"
        ).read_text(encoding="utf-8")
        assert "irreplaceable" not in toml.read_text(encoding="utf-8")

    def test_a_backup_is_not_itself_treated_as_a_model(self, manager, place_gguf):
        """The listing scan globs *.toml; a .toml.bak sitting next to a
        definition must not turn into a second model."""
        place_gguf("Local/model.gguf")
        manager.rebuild_repository()
        manager.rebuild_repository(force=True)

        assert [m["id"] for m in manager._list_local_models_sync()] == ["model"]

    def test_no_backup_appears_when_nothing_was_replaced(self, manager, place_gguf):
        place_gguf("Local/model.gguf")

        manager.rebuild_repository(force=True)

        assert list(manager.local_models_dir.glob("*.bak")) == []


class TestTheCommandItself:
    def test_the_subcommand_is_registered(self):
        args = build_parser().parse_args(["rebuildrepo"])

        assert args.dryrun is False
        assert args.force is False

    def test_the_flags_parse(self):
        args = build_parser().parse_args(["rebuildrepo", "--dryrun", "--force"])

        assert (args.dryrun, args.force) == (True, True)

    def test_no_subcommand_still_means_run_the_server(self, monkeypatch):
        """The entry point behaved that way before it had subcommands, and
        anything invoking it plainly must keep working."""
        called = []
        monkeypatch.setattr("tllama.main.start_server", lambda: called.append(True))

        cli_main([])

        assert called == [True]

    def test_an_overwrite_is_printed_apart_from_the_creations(
        self, manager, place_gguf, monkeypatch, capsys
    ):
        """Forty creations and three replacements in one list means the
        three that cannot be undone are the three least likely to be
        read."""
        place_gguf("Local/already.gguf")
        manager.rebuild_repository()
        place_gguf("Local/fresh.gguf")

        monkeypatch.setattr("tllama.backend.model_manager", manager)
        cli_main(["rebuildrepo", "--dryrun", "--force"])

        out = capsys.readouterr().out
        assert "would create: Local/fresh.toml" in out
        assert "WOULD OVERWRITE: Local/already.toml" in out

    def test_a_dry_run_says_so_at_the_end(
        self, manager, place_gguf, monkeypatch, capsys
    ):
        place_gguf("Local/model.gguf")
        monkeypatch.setattr("tllama.backend.model_manager", manager)

        cli_main(["rebuildrepo", "--dryrun"])

        assert "Nothing was written." in capsys.readouterr().out
