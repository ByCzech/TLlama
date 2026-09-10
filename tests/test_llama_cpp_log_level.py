"""What llama.cpp says during a load, and what it says the rest of the time.

Loading needs the whole of it: every figure /api/ps reports is parsed out
of that output, so verbose stays on and the level filter stays open. What
follows a load is a different matter. llama.cpp keeps talking through the
same callback for the life of the process -- per-request timings that the
API already carries, slot bookkeeping, and mtmd's add_text, which prints
the rendered prompt of every image request at DEBUG. On a server that may
be answering several people, that is a privacy question rather than a
tidiness one, and nobody turned it on.

The filter is the seam, not verbose. verbose doubles as the disable flag
for suppress_stdout_stderr, which dup2s file descriptors 1 and 2 to
/dev/null process-wide; turning it off would black out the server's own
stderr in the middle of unrelated requests, including the account of a
failed load. Moving the level moves nothing but a comparison, and ERROR
and WARN still pass it.
"""

import logging

import asyncio
import threading

import pytest

from llama_cpp import _logger as library_logging


class FakeLlama:
    def __init__(self, name, n_ctx=4096):
        self.name = name
        self._n_ctx = n_ctx

    def n_ctx(self):
        return self._n_ctx


@pytest.fixture
def library_level():
    """Restore whatever the level was, whichever way a test moves it."""
    before = library_logging.logger.level
    try:
        yield library_logging.logger
    finally:
        library_logging.logger.setLevel(before)


class TestTheLevelItChooses:
    def test_it_leaves_errors_and_warnings_able_to_pass(self, library_level, caplog):
        """The callback prints when logger.level <= the line's level.

        WARNING is the highest setting that still lets a WARN line
        through, so it is the quietest one that cannot lose an account of
        something going wrong.
        """
        from tllama.backend import _match_llama_cpp_log_level_to_ours

        caplog.set_level(logging.INFO, logger="tllama.backend")
        _match_llama_cpp_log_level_to_ours()

        assert library_level.level <= logging.WARNING
        assert library_level.level > logging.INFO

    def test_debug_logging_turns_everything_back_on(self, library_level, caplog):
        """TLLAMA_DEBUG reaches this through TLlama's own logger level,
        the same route the captured load log already takes."""
        from tllama.backend import _match_llama_cpp_log_level_to_ours

        caplog.set_level(logging.DEBUG, logger="tllama.backend")
        _match_llama_cpp_log_level_to_ours()

        assert library_level.level == logging.DEBUG


@pytest.fixture
def loadable(manager, gguf_file, toml_file, monkeypatch):
    state = {"hold": None}

    def make(name):
        gguf_file(f"Local/{name}.gguf")
        toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

    def load(model_path, requested_n_ctx, virtual_spec=None):
        name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        # Llama.__init__ calls set_verbose(True) itself, so every load
        # opens the filter again whatever it was set to before.
        library_logging.set_verbose(True)
        if state["hold"] is not None:
            state["hold"].wait(timeout=5)
        return FakeLlama(name, requested_n_ctx or 4096), {}, ""

    monkeypatch.setattr(manager, "_load_model_sync", load)
    state["make"] = make
    return state


class TestWhenItHappens:
    async def test_a_finished_load_closes_the_filter_behind_it(
        self, manager, loadable, library_level, caplog
    ):
        caplog.set_level(logging.INFO, logger="tllama.backend")
        loadable["make"]("alpha")

        await manager.get_model("alpha")

        assert library_level.level == logging.WARNING

    async def test_a_failed_load_closes_it_too(
        self, manager, gguf_file, toml_file, monkeypatch, library_level, caplog
    ):
        """The refusal path pops from _loading as well, and a load that
        threw leaves the filter just as open as one that worked."""
        caplog.set_level(logging.INFO, logger="tllama.backend")
        gguf_file("Local/alpha.gguf")
        toml_file("Local/alpha.toml", '[llm]\nmodel = "Local/alpha.gguf"\n')

        def load(model_path, requested_n_ctx, virtual_spec=None):
            library_logging.set_verbose(True)
            raise RuntimeError("no")

        monkeypatch.setattr(manager, "_load_model_sync", load)

        with pytest.raises(RuntimeError):
            await manager.get_model("alpha")

        assert library_level.level == logging.WARNING

    async def test_it_waits_for_a_load_still_running(
        self, make_manager, gguf_file, toml_file, monkeypatch, library_level, caplog
    ):
        """Closing the filter under a load in flight would empty its
        accounting: every figure /api/ps reports comes out of the output
        that filter is passing, and nothing would say it went missing.
        """
        caplog.set_level(logging.INFO, logger="tllama.backend")
        manager = make_manager(max_loaded_models=2)
        for name in ("alpha", "beta"):
            gguf_file(f"Local/{name}.gguf")
            toml_file(f"Local/{name}.toml", f'[llm]\nmodel = "Local/{name}.gguf"\n')

        hold = threading.Event()
        beta_started = threading.Event()
        seen_by_beta = {}

        def load(model_path, requested_n_ctx, virtual_spec=None):
            name = model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
            library_logging.set_verbose(True)
            if name == "beta":
                beta_started.set()
                hold.wait(timeout=5)
                # alpha has finished by now; if it lowered the level, the
                # rest of beta's own load log is being dropped.
                seen_by_beta["level"] = library_logging.logger.level
            return FakeLlama(name), {}, ""

        monkeypatch.setattr(manager, "_load_model_sync", load)

        beta = asyncio.create_task(manager.get_model("beta"))
        await asyncio.to_thread(beta_started.wait, 5)
        await manager.get_model("alpha")

        hold.set()
        await beta

        assert seen_by_beta["level"] == logging.DEBUG
        assert library_logging.logger.level == logging.WARNING
