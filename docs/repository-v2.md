# Repository format 2: implementation contract

This document describes the agreed repository transition. The name codec and
data-group path calculation are implemented first as shared primitives; they
do not yet switch the daemon's storage format. Do not rename a live repository
by hand. Existing repositories will use the external migration tool introduced
with the storage switch.

## Definitions and data

Public model names remain unchanged. Their one/two/three components select
`Local`, `TLlama` and `HuggingFace` definition areas, respectively. A shared
codec maps each component to a readable portable filename. It escapes
`<>:"|?*%`, terminal dots/spaces and the initial character of a reserved Windows
basename. The actual filename includes `.toml` before encoding. Ordinary
Unicode, internal spaces and `+` remain literal. Percent escapes use uppercase
hexadecimal and decode exactly once. Paths inside TOML are literal physical
repository-relative POSIX paths, not URL-encoded logical names to decode.

Case/Unicode-normalization collisions must be rejected on every platform.
Normalization is only used for this conservative check, never to rewrite a
logical name. Encoded components are limited to 255 UTF-8 bytes and 255 UTF-16
code units; full path limits also depend on the repository root and host APIs.

Every complete group of weights lives under `data/<group-sha256>/`. A single
GGUF uses `weights.gguf`; shards use `weights-00001-of-NNNNN.gguf` and its
ordered siblings. A TOML references the first shard. Projectors and adapters
are independent groups so different models/quantizations can share them.

The directory hash of one file is its content SHA-256. Multiple files use the
existing versioned `tllama-gguf-shards-v1` canonical ordered-digest calculation
from model identity. Original filenames, source, aliases and TOML parameters
do not enter a data-group identity. Public model identity remains a separate
concept, combining these components with non-default parameters.

Group publication must validate complete GGUF headers and stable file hashes,
then publish atomically. Published groups are immutable. Merely calculating
paths is not validation or publication.

## Transition and ownership

Format 2 will be explicitly recorded under `.tllama/`. The daemon will have
one current format; it must not infer encoding from `%HH` or automatically
migrate an old store on startup. Local files enter through `create`; remote
files through `pull`. Remove `rebuildrepo` and old layout fallback code when
the storage switch is delivered.

The one-time converter belongs in `tools/`, outside the Python wheel, and in
the Debian documentation package. It must preflight definitions/collisions,
default to a read-only plan, and apply only on explicit request with daemons
stopped. Preserve TOML comments, backups, file times and hard links; record a
recoverable migration journal. Move unreferenced weights without registering
new models. Remap `[llm].model`, `[mmproj].model` and `runtime.lora_path`.

Valid metadata schema 9 and digest schema 2 caches can keep their payloads:
rewrite the relative `model.path`, rename the path-hashed cache entry and
preserve its file mtime matching the GGUF. Drop invalid entries individually.
Changing paths alone does not require a cache schema bump or rehashing.
Missing content digests require reading the weights once to address the group.

## Removal and maintenance

Ordinary `rm` will delete the definition and reclaim its now-unused groups,
including their shards and caches, without an extra flag. Protect all other
definitions, active/loading/retired workers and in-progress publications.
Coordinate mutations across all daemons sharing a store, not only tasks in
one Python process. If an unreadable definition prevents proving a group is
unused, keep the group and report the skipped cleanup. Apply the same targeted
cleanup after `create --force` or replacing a copy destination.

Whole-store `repo prune` is separate, with a preview and explicit apply.
`repo check` checks definitions, groups and optionally full hashes;
`repo inspect`/`repo stats` explain sharing, orphans and reclaimable data.
These are TLlama extensions with clear unsupported-server errors.

Basic `rm` must also work against Ollama: accept a successful empty HTTP
response, warn once per command on stderr that the server is probably Ollama,
preserve a successful exit status, and keep JSON stdout parseable. An empty
response is a clue, not server identification. Nonempty malformed responses
and HTTP failures remain errors.
