"""Human chat rendering without changing protocol events or conversation data."""

import json
import os
import sys


class ReplyText:
    def __init__(self):
        self.thinking_open = False
        self.answer_open = False
        self.thinking_newline = False
        self.color = sys.stderr.isatty() and os.environ.get('TERM') != 'dumb' and 'NO_COLOR' not in os.environ

    def thinking(self, text):
        if not text:
            return
        if not self.thinking_open:
            if self.answer_open:
                print(flush=True)
                self.answer_open = False
            print(('\x1b[90m' if self.color else '') + 'Thinking:', file=sys.stderr, flush=True)
            self.thinking_open = True
        print(text, end='', file=sys.stderr, flush=True)
        self.thinking_newline = text.endswith('\n')

    def _end_thinking(self):
        if not self.thinking_open:
            return False
        print(('\x1b[0m' if self.color else '') + ('\n' if self.thinking_newline else '\n\n'),
              end='', file=sys.stderr, flush=True)
        self.thinking_open = False
        return True

    def answer(self, text):
        if not text:
            return
        if self._end_thinking():
            print('Answer:', file=sys.stderr, flush=True)
        print(text, end='', flush=True)
        self.answer_open = True

    def tools(self, calls):
        if not calls:
            return
        self._end_thinking()
        if self.answer_open:
            print(flush=True)
            self.answer_open = False
        print('Tool calls: ' + json.dumps(calls, ensure_ascii=False), file=sys.stderr, flush=True)

    def close(self):
        # Always restore terminal color, including failed/cancelled replies
        # that end while reasoning. Do not put styles into saved messages.
        self._end_thinking()
        if self.answer_open:
            print(flush=True)
            self.answer_open = False
