"""Opt-in terminal notifications, independent of response rendering."""

import sys

EVENTS = frozenset({'model_ready', 'response_finished', 'pull_finished', 'create_finished', 'error'})
DEFAULT_EVENTS = ('model_ready', 'response_finished')


class BellNotifier:
    def __init__(self, *, enabled=False, events=DEFAULT_EVENTS):
        self.enabled, self.events = enabled, frozenset(events)

    @classmethod
    def from_args(cls, args):
        return cls(enabled=getattr(args, 'bell', False), events=getattr(args, 'bell_events', DEFAULT_EVENTS))

    def notify(self, event):
        if not self.enabled or event not in self.events:
            return
        try:
            if sys.stderr.isatty():
                sys.stdout.flush()
                sys.stderr.write('\a')
                sys.stderr.flush()
        except (OSError, ValueError, UnicodeError, AttributeError):
            # A notification must never turn a successful request into failure.
            pass
