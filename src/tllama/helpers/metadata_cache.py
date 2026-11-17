from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from tllama.helpers.repository_files import repository_tempfile


SCHEMA_VERSION = 9
"""Layout version of the GGUF metadata cache document.

0235's optional internal _draft summary does not invalidate v9. Ordinary
metadata consumers do not require it. Import checks known-family headers
when their summary is absent and enriches only those entries; unknown
active helper metadata still reaches the authoritative native loader.
The independent content-digest cache is never invalidated by this enrichment.

Bumped to 9 for portable ID comparison against GGUF RoPE defaults.

Bumped to 8 for GGUF license metadata; content digest caches remain valid.
Bumped to 7 for Decision templates, head limits, token IDs and calibration.
Bumped to 6 for explicit embedding output width, distinct from hidden width.
Bumped to 5 for embedding pooling/dimensions, decision type and compact vocab size.
Bumped to 4 for the named tool-use template. Digest caches remain valid.
Bumped to 3 for the compact FIM inventory.

Bumped to 2 when build_model_metadata_payload() gained is_projector,
the shard position and recommended_sampling. A cached document written
before that lacks those keys entirely, and its fingerprint would not
change (the .gguf itself is untouched), so without a bump the new keys
would silently never appear for any model already scanned.
"""

DIGEST_SCHEMA_VERSION = 2
"""Layout version of the content digest cache document.

Version 2 requires a stable before/after file fingerprint during hashing.
Version 1 could label old bytes with a file's post-hash replacement timestamp.

Kept separate from SCHEMA_VERSION on purpose. Metadata is cheap to rebuild
from a GGUF header, the content digest costs a full read of a multi-gigabyte
file. Bumping one must never invalidate the other.
"""

_METADATA_SECTION = "metadata"
_DIGEST_SECTION = "digest"
CACHE_FORMAT_VERSION = 2
"""Cache file mtime carries the model mtime; no model.mtime_ns in JSON."""


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _relative_path(cache_dir: Path, model_path: Path, repository=None) -> str:
    """Identify weights within <repository>/.tllama/metadata-cache.

    A store supplies its physical anchor once; standalone calls derive it
    from the cache's repository parent. Resolve weights for containment.
    POSIX separators keep the identity portable.
    """
    if repository is not None:
        return repository.relative_path(model_path).as_posix()
    root = cache_dir.parent.parent.resolve()
    return model_path.resolve().relative_to(root).as_posix()


def _cache_path(cache_dir: Path, relative: str, section: str) -> Path:
    key = sha256(relative.encode("utf-8")).hexdigest()
    suffix = '.digest.json' if section == _DIGEST_SECTION else '.json'
    return cache_dir / (key + suffix)


def get_metadata_cache_path(cache_dir: Path, model_path: str | Path, *, repository=None) -> Path:
    """Return the JSON metadata cache file path for a model file."""
    path = Path(model_path)
    return _cache_path(cache_dir, _relative_path(cache_dir, path, repository), _METADATA_SECTION)


def get_digest_cache_path(cache_dir: Path, model_path: str | Path, *, repository=None) -> Path:
    """Return the JSON content digest cache file path for a model file."""
    path = Path(model_path)
    return _cache_path(cache_dir, _relative_path(cache_dir, path, repository), _DIGEST_SECTION)


def build_model_file_fingerprint(model_path: str | Path, *, repository=None,
                                 relative_path: str | None = None) -> Dict[str, Any]:
    """Capture a change key, relative when a repository/reference is supplied.

    relative_path is an already checked reference within the caller's store,
    allowing a cache lookup to share it with its filename calculation.
    Standalone callers retain the historical absolute-path snapshot.
    """
    path = Path(model_path)
    stats = path.stat()
    if relative_path is None:
        relative_path = (repository.relative_path(path).as_posix() if repository is not None
                         else str(path.resolve()))

    return {
        "path": relative_path,
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _is_valid_cache_document(
    document: Any,
    fingerprint: Dict[str, Any],
    schema_version: int,
    cache_mtime_ns: int,
) -> bool:
    """Check whether a cache document matches the current model file."""
    if not isinstance(document, dict):
        return False

    if document.get("schema_version") != schema_version:
        return False

    if document.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return False

    model = document.get("model")
    if not isinstance(model, dict):
        return False

    return (
        model.get("path") == fingerprint["path"]
        and model.get("size") == fingerprint["size"]
        and cache_mtime_ns == fingerprint["mtime_ns"]
    )


def _load_cache_document(
    cache_dir: Path,
    model_path: Path,
    schema_version: int,
    section: str,
    repository=None,
) -> Optional[Dict[str, Any]]:
    """
    Load one section of a cache document if it exists and is still valid.

    Invalid, stale or unreadable cache files are treated as cache misses.
    """
    try:
        relative = _relative_path(cache_dir, model_path, repository)
        fingerprint = build_model_file_fingerprint(model_path, relative_path=relative)
        cache_path = _cache_path(cache_dir, relative, section)
        with cache_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
            # Read attributes of the same inode as the JSON, even if another
            # daemon atomically replaces the cache path during this read.
            cache_mtime_ns = os.fstat(handle.fileno()).st_mtime_ns

        if not _is_valid_cache_document(document, fingerprint, schema_version, cache_mtime_ns):
            return None

        payload = document.get(section)
        if not isinstance(payload, dict):
            return None

        return payload
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _save_cache_document(
    cache_dir: Path,
    model_name: str,
    model_path: Path,
    schema_version: int,
    section: str,
    payload: Dict[str, Any],
    fingerprint: Optional[Dict[str, Any]] = None,
    repository=None,
) -> None:
    """
    Save one cache document using an atomic replace.

    Cache write failures are intentionally non-fatal for the application.
    """
    try:
        if repository is not None:
            from tllama.repository_runtime import read_format
            relative = repository.relative_path(model_path)
            if (read_format(repository) is not None
                    and (relative.parts[:1] == ('data',)
                         or relative.parts[:2] == ('.tllama', 'uploads'))):
                # Serialize the final cache publication with physical GC.
                # A completed, delayed scan must not resurrect an orphan's
                # cache after its weights and previous cache were removed.
                with repository.lock():
                    if not model_path.is_file():
                        return
                    return _save_cache_document(cache_dir, model_name, model_path,
                        schema_version, section, payload, fingerprint=fingerprint)
        relative = _relative_path(cache_dir, model_path, repository)
        cache_path = _cache_path(cache_dir, relative, section)

        if fingerprint is None:
            fingerprint = build_model_file_fingerprint(model_path, relative_path=relative)

        # Preserve the captured size/mtime and path: delayed writes must not
        # label an old scan with the current file's identity.
        captured = Path(fingerprint["path"])
        if captured.is_absolute():
            root = repository.physical_root if repository is not None else cache_dir.parent.parent.resolve()
            captured = captured.relative_to(root)
        if '..' in captured.parts or captured.as_posix() != relative:
            return
        fingerprint = {**fingerprint, "path": captured.as_posix()}
        cache_dir.mkdir(parents=True, exist_ok=True)

        document = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "schema_version": schema_version,
            "created_at": _utc_now_iso(),
            "model": {
                "name": model_name,
                "path": fingerprint["path"],
                "size": fingerprint["size"],
            },
            section: payload,
        }

        fd, tmp_name = repository_tempfile(
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            directory=cache_dir,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    document,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")

            # Set the captured model timestamp after the final write and before
            # publication. Never stat the model again here: a delayed old scan
            # must not acquire a replacement model's newer timestamp.
            os.utime(tmp_name, ns=(fingerprint["mtime_ns"], fingerprint["mtime_ns"]))
            os.replace(tmp_name, cache_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception:
        return


def load_metadata_cache(
    cache_dir: str | Path,
    model_path: str | Path,
    *,
    repository=None,
) -> Optional[Dict[str, Any]]:
    """Load cached GGUF metadata if the cache file exists and is still valid."""
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    return _load_cache_document(
        cache_dir=cache_dir,
        model_path=model_path,
        schema_version=SCHEMA_VERSION,
        section=_METADATA_SECTION,
        repository=repository,
    )


def save_metadata_cache(
    cache_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    metadata: Dict[str, Any],
    *,
    fingerprint: Optional[Dict[str, Any]] = None,
    repository=None,
) -> None:
    """Save GGUF metadata with the fingerprint of the file actually scanned.

    A supplied fingerprint prevents a delayed write from labeling old metadata
    with the identity of weights replaced after the scan. The cache loader will
    reject that old snapshot for the new file.
    """
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    _save_cache_document(
        cache_dir=cache_dir,
        model_name=model_name,
        model_path=model_path,
        schema_version=SCHEMA_VERSION,
        section=_METADATA_SECTION,
        payload=metadata,
        fingerprint=fingerprint,
        repository=repository,
    )


def load_digest_cache(
    cache_dir: str | Path,
    model_path: str | Path,
    *,
    repository=None,
) -> Optional[Dict[str, Any]]:
    """Load the cached content digest if the cache file exists and is still valid."""
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    return _load_cache_document(
        cache_dir=cache_dir,
        model_path=model_path,
        schema_version=DIGEST_SCHEMA_VERSION,
        section=_DIGEST_SECTION,
        repository=repository,
    )


def save_digest_cache(
    cache_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    digest: Dict[str, Any],
    *,
    fingerprint: Optional[Dict[str, Any]] = None,
    repository=None,
) -> None:
    """Save a model content digest to its JSON cache file."""
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    _save_cache_document(
        cache_dir=cache_dir,
        model_name=model_name,
        model_path=model_path,
        schema_version=DIGEST_SCHEMA_VERSION,
        section=_DIGEST_SECTION,
        payload=digest,
        fingerprint=fingerprint,
        repository=repository,
    )
