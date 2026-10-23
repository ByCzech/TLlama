"""Transient terminal feedback; redirected and JSON output stays quiet."""

import asyncio
import os
import sys

FRAMES = '⣷⣯⣟⡿⢿⣻⣽⣾'
INTERVAL = 0.1


class WaitStatus:
    def __init__(self, label, *, enabled=True):
        self.output = sys.stderr
        self.label = label
        self.enabled = enabled and self.output.isatty() and os.environ.get('TERM') != 'dumb'
        if self.enabled:
            try:
                FRAMES.encode(getattr(self.output, 'encoding', None) or 'utf-8')
            except UnicodeEncodeError:
                self.enabled = False
        self.task = None
        self.drawn = False
        self.stopped = False

    async def __aenter__(self):
        if self.enabled:
            self.task = asyncio.create_task(self._animate())
        return self

    async def _animate(self):
        index = 0
        while not self.stopped:
            self.output.write(f'\r{FRAMES[index % len(FRAMES)]} {self.label}')
            self.output.flush()
            self.drawn = True
            index += 1
            await asyncio.sleep(INTERVAL)

    def stop(self):
        # Synchronous clearing ensures the next output chunk cannot race
        # with the animation. Joining its task remains mandatory on exit.
        self.stopped = True
        if self.task is not None:
            self.task.cancel()
        if self.drawn:
            self.output.write('\r' + ' ' * (len(self.label) + 2) + '\r')
            self.output.flush()
            self.drawn = False

    async def __aexit__(self, *args):
        self.stop()
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
