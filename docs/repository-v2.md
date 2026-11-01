# Repository format 2: implementation contract

The name codec, transactional data-group store, offline converter and ordinary
format-2 runtime are implemented. See [model-repositories.md](model-repositories.md)
for current usage and [the converter](../tools/README.md) for upgrades.
Physical reclamation and whole-store maintenance remain the next stage.

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

Format 2 is explicitly recorded in `.tllama/repository.json`. The daemon has
one current format; it never infers encoding from `%HH` or automatically
converts an old store. Empty stores initialize automatically; populated unmarked
stores require the offline converter. Local files enter through `create`;
remote files through `pull`. Definition mutations and group publication use the
shared repository lock. `rebuildrepo` and physical-file-derived model naming
have been removed. Pulls stage complete groups privately, then register a
portable definition and source association; a repeated pull preserves edited
settings and stable aliases. No raw GGUF scan creates definitions.

The one-time converter is `tools/migrate-repository-v2.py`, outside the Python
wheel. Its Debian documentation installation is a separate packaging patch.
It preflights definitions/collisions,
defaults to a read-only plan, and applies only on explicit request with daemons
stopped. Preserve TOML comments, backups, file times and hard links; record a
recoverable migration journal. Move unreferenced weights without registering
new models. Remap `[llm].model`, `[mmproj].model` and `runtime.lora_path`.

Valid metadata schema 9 and digest schema 2 caches can keep their payloads:
rewrite the relative `model.path`, rename the path-hashed cache entry and
preserve its file mtime matching the GGUF. Drop invalid entries individually.
Changing paths alone does not require a cache schema bump or rehashing.
Missing content digests require reading the weights once to address the group.
Migration must also accept repositories from releases older than 1.3.1;
upgrading through 1.3.1 first is not required. Reuse only independently valid,
recognized cache documents. Old/unknown schemas, legacy absolute-path or JSON
mtime formats, corrupt JSON and stale timestamps are cache misses, not reasons
to abort migration. Drop obsolete cache entries rather than carrying old cache
reader compatibility into the new runtime. A remotely claimed digest (source
`hf`) requires local measurement before content-addressed publication.

## Implemented data publication layer

`repository_store.prepare_group` validates complete ordered GGUF headers,
captures source signatures, and obtains actual local content hashes. Current
metadata payloads and locally measured digests may be reused from cache;
unrecognized caches are ignored. Preparation is read-only, including caches.

`RepositoryStore.publish` stages a whole group and atomically renames its
directory under `data/`. By default it copies into independent files and
verifies their bytes while copying. Trusted immutable repository-owned upload
or offline migration inputs may instead use `link_owned=True` to preserve
hard links without copying gigabytes. Inputs are not removed by publication:
the offline migration journal must own any subsequent source removal.
Source changes and incomplete, corrupt or symlinked existing groups are errors;
existing groups are verified and reused, never overwritten. Successful
publication warms the current metadata/digest caches at the final file paths.

The store uses an exclusive OS-backed `.tllama/repository.lock`, reentrant
within one store/thread and released on process exit. Windows uses a byte-range
lock; POSIX uses flock. The backing shared filesystem must honor locks across
all participating hosts. Locking a publication does not by itself implement
worker pins, definition transactions or garbage collection: those are required
before runtime physical deletion is enabled. Definition transactions now use
this lock too; downloads/uploads stage bytes before publication.

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
