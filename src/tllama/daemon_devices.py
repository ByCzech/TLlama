"""One-shot native device enumeration, without importing the application."""

import os
from pathlib import Path
import sys
import tempfile


def list_devices(executable):
    from tllama.device_inventory import discovery_environment, run_probe
    with tempfile.TemporaryDirectory(prefix='tllama-list-devices-') as directory:
        result = run_probe(Path(executable), '--list-devices', discovery_environment(os.environ, {}), directory)
    for stream, data in ((sys.stdout, result.stdout), (sys.stderr, result.stderr)):
        if hasattr(stream, 'buffer'):
            stream.buffer.write(data)
            stream.buffer.flush()
        else:
            stream.write(data.decode('utf-8', errors='replace'))
            stream.flush()
    if result.outcome != 'exited' or not result.reaped:
        print('tllamad: Device listing failed: ' + (result.error or result.outcome), file=sys.stderr)
        return 1
    return result.returncode if result.returncode >= 0 else 128 - result.returncode
