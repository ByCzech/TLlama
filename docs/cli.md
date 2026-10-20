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
