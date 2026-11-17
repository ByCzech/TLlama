# Repository startup baseline

Run from a checkout with its Python dependencies. This local diagnostic measures
the repository code before and after path/cache optimizations. It does not load
a native worker, contact a daemon or discover a daemon configuration file.
Use `--config /etc/tllama/server.toml` when the definition preparation should
include those explicit global settings; `--models-dir` always selects the root.

## Capture

Use one existing main-model name; `--model` can be repeated. Each output must be
a new file. Run as the account normally accessing this repository, for example
the `tllama` service account when appropriate for the local installation.

```sh
PYTHONPATH=src python3 scripts/repository-startup-profile.py \
  --models-dir /var/lib/tllama/models --model 'YOUR_MAIN_MODEL' \
  --repeat 3 --output /tmp/tllama-startup-time.json

PYTHONPATH=src python3 scripts/repository-startup-profile.py \
  --models-dir /var/lib/tllama/models --model 'YOUR_MAIN_MODEL' \
  --repeat 3 --trace --output /tmp/tllama-startup-counts.json
```

The ordinary first listing may populate missing or invalid metadata/digest
caches, including hashing weights with no valid digest. No cache is deleted or
invalidated by the diagnostic. With valid warm caches, the counts show whether
header reads, full hashes or cache-save attempts still occur. Keep the same
weights, definitions, configuration and context when comparing before/after.

The script preserves the configured root spelling, including an alias. For an
alias/direct-root comparison, repeat the commands with the other root spelling
and new outputs. Current application path handling remains unchanged by this
diagnostic. Do not switch a root symlink underneath a running process.

## Interpretation

- `import_seconds`: importing the backend and diagnostic dependencies in this
  process; not the CLI import time or the complete Python process startup.
- `configuration`, `manager`: building the selected configuration and manager.
- `list:1..N`: the actual enriched listing used by the API, first with fresh
  manager RAM state, then repeatedly in the same manager. Existing disk cache
  is preserved; the OS page cache is not flushed or declared cold.
- `inventory`: the simpler definition listing, measured after the full listing.
- `metadata:MODEL:N`: metadata lookup after the listing has warmed RAM state.
- `prepare:MODEL:N`: resolving the definition, artifacts, content identity and
  load parameters; no residency admission, pins, quota wait or native load.

Only compare latency using the run without `--trace`. Trace timing includes
counter overhead. Its per-phase counts cover backend TOML parsing, GGUF header
reader calls, full-file hash calls, metadata/digest loads and hit/miss outcomes,
cache-save attempts, backend fingerprint calls and Python `Path.stat/resolve`.
They are selected Python call counts, not an exhaustive syscall/byte tracer.
In particular, save attempts are not proof of completed disk writes and
`path.stat` excludes direct `os.stat/lstat` calls. Tracing works in the existing
I/O threads, is process-local and is removed on completion or failure.

## Separate client, HTTP and native load

Use the same daemon and repeat these commands several times. The local tool's
manager has independent RAM state and does not attach to that daemon.

```sh
URL=http://127.0.0.1:54800
/usr/bin/time -f 'cli_version_seconds=%e' tllama --version >/dev/null
/usr/bin/time -f 'cli_list_seconds=%e' tllama --host "$URL" list >/dev/null
curl --fail --silent --show-error --output /dev/null \
  --write-out 'http_tags_seconds=%{time_total}\n' "$URL/api/tags"
```

CLI list includes client configuration/imports, HTTP, server listing and output
formatting. Direct HTTP excludes the TLlama CLI but includes the server work.
These runs are sequential samples, not exact subtractable components.

For an already resident main model, repeat the following command twice and
retain both complete statistics blocks. It deliberately performs a real
request; use the same context and options in both samples.

```sh
tllama --host "$URL" run 'YOUR_MAIN_MODEL' \
  --option num_ctx=4096 --option num_predict=1 --verbose 'Hello'
```

Retain `tllama ps --json` before and after to establish residency. A first load
or configuration-triggered reload belongs in its own sample. The local
`prepare` time alone is not the daemon's entire `Server preparation` time and
does not measure first-token latency or disk-to-GPU loading. Do not attribute
a long first load to cache overhead without this separation.

## Regression checks

```sh
pytest -q tests/test_repository_startup_profile.py \
  tests/test_repository_cache_identity.py tests/test_cache_file_timestamps.py \
  tests/test_virtual_model_metadata.py tests/test_model_artifacts.py
```

The diagnostic tests use small real GGUF headers with shared definitions and
shards. They check cold work, warm cache reuse through a root alias, unchanged
cache/source bytes and mtimes, no worker start, errors and output protection.
These small fixtures establish counter behavior, not real repository latency.
