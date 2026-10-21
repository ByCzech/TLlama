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
Without `--timeout`, list/show/pull/run have no response-read deadline: rebuilding
metadata or waiting for a server-side download lookup may take longer than a
fixed limit. Other commands retain a 30-second HTTP inactivity timeout.
Connection setup remains capped at 10 seconds, including for list/show/pull/run.
`--timeout SECONDS` sets a positive finite inactivity limit explicitly, including
response reads; shorter values also lower the connection deadline. This is not
an overall command duration limit. Proxy and CA environment settings follow
HTTPX; an optional proxy dependency must be installed if that proxy needs it.
There is no automatic retry, server startup or model download. A connection
failure tells the user which server to start or select.

Command results go to stdout; diagnostics and human progress go to stderr. Exit status is 0 for
success/help, 1 for server/connection/protocol failures, 2 for usage/client
configuration errors and 130 for interruption. `--version` is entirely local;
the `version` command contacts the server and fails if it cannot be reached.

`--json` preserves the full server response. `show` text selectors cannot be
combined with `--json`; they write the field verbatim, including any final
newline. `show --modelfile` returns TOML, not an Ollama Modelfile conversion.
`ps` derives processor placement from the server's compatibility size fields;
unknown values remain unknown. These fields are not a measurement of physical
RAM/VRAM residency. Detailed memory/provenance remains available with `ps --json`.

## Cold metadata and waiting

For a human-readable list/show request taking more than two seconds, the client
prints a single waiting notice to stderr, then keeps waiting. JSON output has no
waiting notice. Ctrl+C cancels and drains the client's pending HTTP request.
Connect, response-read, request-write and connection-pool timeouts have distinct
messages. A response timeout does not establish that the server has stopped.

The daemon logs start/completion of actual GGUF metadata reads and model SHA-256
computation, including elapsed time. Cache hits do not produce those messages.
The first listing can trigger this work; startup does not eagerly index models.

Patch 0083 raised the metadata-cache revision from 7 to 8 because the extracted
payload now includes general.license, general.license.name and general.license.link.
The JSON envelope layout did not change, but old payloads do not contain those
fields and must be refreshed. Digest-cache revision remains 1; this metadata
refresh alone does not cause a full reread of weight contents for hashing.
Existing valid persistent caches survive a daemon restart.

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
The client completes cleanup of the HTTP decoding chain before returning,
including success, malformed progress and early termination.

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

Server TOML/daemon switches and systemd packaging remain
subsequent planned work. The client layer uses async HTTPX operations so that
later REPL support can share their validation and response ownership.

The Python client layer now provides `TLlamaClient.chat(model, messages, ...)`
as an async iterator over original `/api/chat` events. It preserves content,
thinking, tool calls and terminal statistics for a caller to render/store.
Optional `options`, `think`, `format` and `keep_alive` controls are omitted when
unspecified, preserving server/model defaults; explicit false and zero values
are sent unchanged. There is no automatic retry or model download.
Use `contextlib.aclosing` when consuming a stream that may end early. Completion
requires a boolean `done: true` event; EOF alone is an error. The shared HTTP
reader closes before returning on completion, errors and cancellation. Chat
reads have no default inactivity deadline during model loading or generation;
an explicit client timeout still applies, and connection setup stays bounded.
This is the transport shared by run and the REPL.

## Run a model

```sh
tllama run MODEL 'Explain this code.'
cat source.py | tllama run MODEL 'Review the code above.'
tllama run MODEL 'Return JSON.' --format json --option temperature=0
cat decision.json | tllama run DECISION_MODEL
```

`run` checks `/api/show` capabilities first. Completion models use streamed
`/api/chat`; embedding models use `/api/embed` and print a JSON result. Decision
models use `/v1/systemone`: input must be a JSON object with exactly `state` and
`questions`, and the result is JSON. The model name comes from the command.
No model is automatically downloaded. Server/model defaults remain in effect
unless a control is supplied.

Prompt arguments are joined with spaces. When stdin is redirected, all of its
text is read as one input and prepended to an argument prompt with two newlines.
Whitespace and slash-prefixed text are otherwise preserved. An explicit empty
prompt is still a request; stdin EOF does not open an interactive terminal.

`--option NAME=JSON` can be repeated (last value for a name wins). `--system`
sets a session system message, `--think` accepts true/false or a supported level,
`--format` accepts json or a JSON schema object, and `--keep-alive` selects a
server residence duration. The server validates parameter names and supported
values. Inappropriate controls for embedding/Decision modes fail explicitly.
`--hide-thinking` affects display only: reasoning normally goes to stderr,
answer text to stdout. Thinking is grouped under `Thinking:` and followed by
an `Answer:` separator before answer text starts. These labels go to stderr,
so redirected stdout still contains only the answer. Reasoning uses the
terminal's gray/bright-black color; `NO_COLOR`, non-TTY stderr and `TERM=dumb`
disable color while retaining labels. Terminal color is restored on normal
completion, errors and cancellation. Hidden thinking has no labels or styles;
the original reasoning text remains in conversation messages. Tool calls are
reported on stderr, never executed.
`--json` emits original chat events as NDJSON, including reasoning and stats;
embedding/Decision output is already JSON. Errors may follow partial streamed
output and make the exit status nonzero; requests are never automatically retried.
Ctrl+C closes the request and exits with status 130 in one-shot mode.

## Interactive run

`tllama run MODEL` opens a prompt_toolkit REPL when stdin is a terminal and
no prompt arguments were supplied. Before the first prompt it prepares the
model through an empty, non-streaming generate request. Runtime options and
keep-alive are forwarded without overriding server defaults. A zero effective
keep-alive retains the existing server unload semantics: the client reports
that preload was not retained, and the real request loads the model later.
One-shot/stdin calls do not add this preliminary request.

On a terminal, preparation displays `Loading model…` with the spinner
`⣷⣯⣟⡿⢿⣻⣽⣾` (100 ms per frame). Each inference request displays
`Waiting for response…` until visible text/tool calls arrive, or the request
completes. With hidden thinking it continues through hidden reasoning. This
general wait label does not claim to identify queueing, loading or prefill.
The spinner clears and its task finishes on success, failure and cancellation.
It writes only to stderr and is disabled for JSON, redirected stderr, dumb
terminals and encodings that cannot display the selected frames.

Enter sends a prompt; Alt+Enter inserts a
newline. Bracketed paste preserves multiline text as one input. Tab completes
slash commands; Up/Down recall input from this process's in-memory history.
No prompts or messages are written to a history file. Prompts and REPL control
messages go to stderr, including when `--json` is selected.

- `/help` or `/?`: list commands and key bindings.
- `/bye` or `/exit`: end the session. Ctrl+D also exits at an empty prompt.
- `/clear`: forget conversation messages, retaining session settings and input
  history. This does not unload the model or clear the server's KV cache.
- `/show model`: fetch model information from the server.
- `/show parameters`: display explicit session request controls, not merged
  model/server defaults. `/show system` shows the session system override, or
  null when the model/server supplies it.
- `/set parameter NAME JSON`: update a request option for future turns.
- `/set system TEXT`: replace the session system prompt and forget prior turns.
- `/set think VALUE`, `/set format json|SCHEMA|off`, `/set keep_alive VALUE`:
  update the corresponding session controls. Format off removes the override.

Settings remain in the session and never rewrite a model definition. The server
validates parameter names/values on the next request. Embedding models accept
their supported runner options; Decision models accept no generation controls.
Embedding/Decision REPL inputs are independent requests, not conversation turns.
Use `//text` to send a prompt beginning with `/text`. Other ordinary prompt text
is passed literally, without shell parsing. Save/load/create/copy are not part
of this initial shell.

Ctrl+C while editing discards current input. During inference it cancels and
closes the active HTTP request, waits for client cleanup and returns to the
prompt. Server disconnect handling remains responsible for stopping native
work; the client cannot promise immediate GPU cancellation. Only completed
chat turns enter conversation history. Errors and cancelled partial replies
are displayed but are not reused as context. There is no automatic retry.

## Verbose timing and usage

`tllama run MODEL --verbose` (or `-v`) prints statistics to stderr after each
completed request. `/set verbose` and `/set quiet` toggle this locally in every
REPL mode; they do not send model options or change stored definitions.

Client total time covers the request from submission through completion.
First-text time is measured at the first nonempty content or thinking chunk,
including hidden thinking; empty stream events do not count. Initial REPL
preload request time is reported separately before the first prompt when
verbose was enabled at startup. One-shot requests have no separate preload.

Where supplied, the summary shows server total/preparation time, evaluated and
cached prompt tokens, generated tokens, prompt/generation durations and tok/s,
and the stop reason. Server preparation is the reported load/admission span;
it is not necessarily time spent reading weights. Rates use the matching
evaluated/generated counts and measured durations, never cached token counts
or total wall time. Missing measurements are omitted. Measured zero counts or
durations remain zero; a zero duration has no calculated rate.

Embedding/Decision results keep their JSON stdout and display only available
statistics. Decision input/output usage is not presented as generation speed.
JSON chat output also stays unchanged: explicit verbose output goes to stderr,
and reasoning styles/spinner frames are never inserted into protocol events.
Incomplete, failed or cancelled replies have no completed-request summary.

## Copy model definitions

`tllama cp SOURCE DESTINATION` uses `POST /api/copy` with source/destination
names from the model list. It copies TOML text, preserving comments and shared
repository weight/projector references. No worker loads and no weights are
copied or downloaded. The new definition is independent of its source.
An existing destination returns HTTP 409; `--force` explicitly replaces it.
Source and destination cannot name the same definition. JSON output reports
the original success object. Definition publication is atomic, including
concurrent creates without force. Forced changes become visible through the
existing definition fingerprint/reload mechanism; current leases retain their
already acquired generation. A lost response is reported without an automatic
retry or a claim that the operation did not complete.

Large GGUF uploads use `HEAD /api/blobs/sha256:HEX` and binary
`POST /api/blobs/sha256:HEX` with `Content-Type: application/octet-stream`.
POST streams bytes to a temporary file, verifies the SHA-256 and basic GGUF
header, then publishes an immutable staging entry. Wrong hashes, incomplete
and interrupted bodies leave no published entry or temporary file. Identical
uploads are idempotent and never replace an existing inode. Uploading alone
does not register a model; create will publish named GGUF files into Local and
a definition only after its validation succeeds. Completed staging entries
are retained for reuse if a subsequent create fails; garbage collection is
separate work. This binary API does not accept multipart/form-data.

## Create and import models

```sh
tllama create MODEL -f model.toml
tllama create MODEL -f model.toml --force
tllama create MODEL -f model.toml --json
```

The file is **TLlama TOML**, read on the client. For a definition that refers to
existing server repository files, create only writes the TOML:

```toml
[llm]
model = "Local/already-present.gguf"
```

`from` instead uploads a client-side GGUF into the server's `Local` storage.
Relative `from` paths are relative to the client's current working directory,
even when the TOML file is elsewhere. Absolute paths and `~` refer to the client
machine. An optional projector is imported the same way:

```toml
[llm]
from = "weights/model.gguf"

[mmproj]
from = "weights/projector.gguf"
```

Use exactly one of model/from in each section. Client files remain untouched.
The server never interprets from as permission to read its own filesystem.
Each source is validated, hashed and uploaded as a bounded stream; already
uploaded digests are reused after HEAD. The staging directory is in Local so
publication needs no second multi-gigabyte copy. Imported files are published
under `Local/imports/FIRST_FILE_SHA256/original-filename.gguf`; from is replaced
by that repository-relative model path in the server TOML, preserving comments.
Definitions and weight/projector files retain their separate names and roles.

For a split GGUF, from must name the first shard. The client automatically
includes the complete sibling set described by split.count/split.no, with the
standard NAME-00001-of-NNNNN.gguf names. Both sides verify shard positions and
counts, and the server preserves filenames together so native shard discovery
works. Missing, mismatched or continuation-only imports fail explicitly.
Each group is limited to 2048 shards. GGUF structural checks do not prove a
model's compatibility with a particular native backend/device.
Verified upload digests and inspected headers warm the existing caches for
the final Local paths; an unchanged imported file is not hashed or scanned
again just to list it. Cache versions are unchanged. Changed staged bytes are
rechecked and cannot be published under their old digest.

Before hashing/uploading, create preflights the name, TOML and references to
existing server files. It reserves no name: final publication still checks
conflicts atomically. Existing definitions require --force; that flag never
replaces weight bytes. A new TOML becomes visible only after all imports and
references validate. Completed blobs/imported weights can remain for reuse
after a late conflict/failure; they are not deleted behind other definitions.
There is no automatic retry, model load, conversion or quantization.

The native TLlama `POST /api/create` JSON contract is model, definition (TOML
text), optional force and uploads. uploads maps llm/mmproj to ordered arrays
of `{ "digest": "sha256:HEX", "filename": "original.gguf" }`, exactly for
sections containing from. Upload every referenced digest through the blob API
first. With no from, omit uploads. Success is `{ "status": "success", "model":
"NAME" }`. A dry_run request without uploads performs the definition/name/ref
preflight and returns status ready; it does not inspect pending local sources
or publish anything. The definition limit is 2 MiB of UTF-8 text.

This create contract does not parse Ollama Modelfile or accept Ollama's full
structured create schema. Unsupported fields fail explicitly. Save/load of
REPL conversations remains separate work.
