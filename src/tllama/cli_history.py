"""Prompt history is independent of conversation messages and server state."""

import sys

from tllama.client_state import state_directory, private_directory, private_file


def input_history(server, model, *, enabled=True, clear=False):
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    class PersistentHistory(FileHistory):
        def store_string(self, value):
            try:
                super().store_string(value)
            except (OSError, UnicodeError) as exc:
                print(f'tllama: Cannot write input history: {exc}', file=sys.stderr)

        def load_history_strings(self):
            try:
                yield from super().load_history_strings()
            except (OSError, UnicodeError) as exc:
                print(f'tllama: Cannot read input history: {exc}', file=sys.stderr)
    if not enabled and not clear:
        return InMemoryHistory()
    path = state_directory(server, model, 'history') / 'input.history'
    try:
        if clear:
            path.unlink(missing_ok=True)
        if not enabled:
            return InMemoryHistory()
        private_directory(path.parent)
        private_file(path)
        return PersistentHistory(str(path))
    except OSError as exc:
        print(f'tllama: Cannot persist input history: {exc}; using memory only.', file=sys.stderr)
        return InMemoryHistory()
