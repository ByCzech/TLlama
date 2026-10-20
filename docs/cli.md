# Client and daemon commands

`tllamad` starts the foreground server. `tllama` is the HTTP client for a local
or remote server; invoking it without a command prints help. It does not start
or import a server, discover an inference runtime, or create a model store.
There are no client `serve` or `start` commands.

```sh
tllamad                         # foreground server; existing TLLAMA_* environment
python -m tllama.daemon          # equivalent from a checkout
python -m tllama.main            # retained explicit ASGI/server module spelling
tllama --help
python -m tllama --help          # client from a checkout
tllama --version                # installed client version; no network access
tllamad --version               # installed daemon version; no server startup
```

Existing environments need their console scripts regenerated after this change
(for example, reinstall the package or its editable installation). A source
checkout can use the module commands immediately. The server TOML and additional
daemon switches remain a separate planned block; the existing server
configuration via environment variables is unchanged here.

## Local repository maintenance

`rebuildrepo` now belongs to the daemon host, not the HTTP client:

```sh
tllamad rebuildrepo --dryrun
tllamad rebuildrepo
tllamad rebuildrepo --force --overwrite-backups
```

This replaces `tllama rebuildrepo`; it operates on the local TLLAMA_MODELS store,
with the existing backup and validation rules. It does not start the server.
See [model-repositories.md](model-repositories.md) for details.

## HTTP client basics

```sh
tllama list                      # alias: ls
tllama show MODEL
tllama show MODEL --modelfile     # original TLlama TOML, verbatim
tllama show MODEL --parameters
tllama show MODEL --template
tllama show MODEL --system
tllama show MODEL --license       # GGUF license identifier when available
tllama ps
tllama version                   # client + connected server version
tllama --host 192.0.2.10:54800 list --json
```

Client address selection is `--host > TLLAMA_HOST > 127.0.0.1:54800`.
HTTP and HTTPS URLs, an optional reverse-proxy path prefix, host:port, :port,
and bracketed/bare IPv6 are supported. An omitted port always means 54800;
write an explicit :443 or :80 when a reverse proxy uses that port. Wildcard
listener addresses 0.0.0.0 and :: connect through loopback. The client preserves
the HTTP(S) scheme; the daemon's legacy environment parser does not enable TLS.
Credentials, query strings and fragments in the address are refused.

`--host`, `--timeout` and `--json` work before or after the subcommand.
`--timeout SECONDS` is a positive finite HTTP inactivity timeout (default 30 s);
connection setup is capped at 10 s. Proxy and CA environment settings follow
HTTPX; an optional proxy dependency must be installed if that proxy needs it.
There is no automatic retry, server startup or model download. A connection
failure tells the user which server to start or select.

Human output goes to stdout; failures go to stderr. Exit status is 0 for
success/help, 1 for server/connection/protocol failures, 2 for usage/client
configuration errors and 130 for interruption. `--version` is entirely local;
the `version` command contacts the server and fails if it cannot be reached.

`--json` preserves the full server response. `show` text selectors cannot be
combined with `--json`; they write the field verbatim, including any final
newline. `show --modelfile` returns TOML, not an Ollama Modelfile conversion.
`ps` derives processor placement from the server's compatibility size fields;
unknown values remain unknown. These fields are not a measurement of physical
RAM/VRAM residency. Detailed memory/provenance remains available with `ps --json`.

## Pull, remove and unload

```sh
tllama pull org/repository/file.gguf
tllama pull org/repository/file.gguf --json
tllama rm first-model second-model
tllama stop MODEL
```

`pull` uses the server's existing Hugging Face reference syntax and credentials;
it is not an Ollama registry download. Human progress goes to stderr, followed
by a success line on stdout. `--json` emits each progress object as one NDJSON
line on stdout. HTTP errors, in-stream errors, invalid progress and a stream
ending without success all fail the command. Ctrl+C closes the client stream;
the server's existing downloader thread is not guaranteed to stop immediately.

`rm` deletes definitions on the selected server and leaves weight files on disk.
All supplied names are attempted even if one fails; any failure makes the exit
status 1. `--json` emits one record per name, with `model` and either the original
`response` object or an `error`. Errors also go to stderr. No operation is
retried automatically, so a lost response is reported without guessing whether
the server completed the deletion.

`stop` sends an empty non-streaming `/api/generate` request with keep_alive=0.
This unload path runs before generation-mode admission, so it also handles
embedding/Decision workers and an already-unloaded name without inference.
Success means the unload was requested; active leases drain according to the
existing lifecycle before the worker exits. It does not forcibly interrupt
other clients' inference. `stop --json` returns the unload response.

`run`, REPL, create/copy, server TOML/daemon switches and systemd packaging remain
subsequent planned work. The client layer uses async HTTPX operations so that
later REPL support can share their validation and response ownership.
