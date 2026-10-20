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
