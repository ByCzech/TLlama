from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from tllama.helpers.repository_files import repository_tempfile


SCHEMA_VERSION = 8
"""Layout version of the GGUF metadata cache document.

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

DIGEST_SCHEMA_VERSION = 1
"""Layout version of the content digest cache document.

Kept separate from SCHEMA_VERSION on purpose. Metadata is cheap to rebuild
from a GGUF header, the content digest costs a full read of a multi-gigabyte
file. Bumping one must never invalidate the other.
"""

_METADATA_SECTION = "metadata"
_DIGEST_SECTION = "digest"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _relative_path(cache_dir: Path, model_path: Path) -> str:
    """Identify weights within <repository>/.tllama/metadata-cache.

    Resolve both paths so a symlink to the repository shares its cache, but
    reject weights outside it. POSIX separators keep the identity portable.
    """
    root = cache_dir.parent.parent.resolve()
    return model_path.resolve().relative_to(root).as_posix()


def _cache_key(cache_dir: Path, model_path: Path) -> str:
    return sha256(_relative_path(cache_dir, model_path).encode("utf-8")).hexdigest()


def get_metadata_cache_path(cache_dir: Path, model_path: str | Path) -> Path:
    """Return the JSON metadata cache file path for a model file."""
    path = Path(model_path)
    return cache_dir / f"{_cache_key(cache_dir, path)}.json"


def get_digest_cache_path(cache_dir: Path, model_path: str | Path) -> Path:
    """Return the JSON content digest cache file path for a model file."""
    path = Path(model_path)
    return cache_dir / f"{_cache_key(cache_dir, path)}.digest.json"


def build_model_file_fingerprint(model_path: str | Path) -> Dict[str, Any]:
    """Build a lightweight model file fingerprint for cache invalidation."""
    path = Path(model_path)
    stats = path.stat()

    return {
        "path": str(path.resolve()),
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _is_valid_cache_document(
    document: Any,
    fingerprint: Dict[str, Any],
    schema_version: int,
) -> bool:
    """Check whether a cache document matches the current model file."""
    if not isinstance(document, dict):
        return False

    if document.get("schema_version") != schema_version:
        return False

    model = document.get("model")
    if not isinstance(model, dict):
        return False

    return (
        model.get("path") == fingerprint["path"]
        and model.get("size") == fingerprint["size"]
        and model.get("mtime_ns") == fingerprint["mtime_ns"]
    )


def _load_cache_document(
    cache_dir: Path,
    model_path: Path,
    schema_version: int,
    section: str,
) -> Optional[Dict[str, Any]]:
    """
    Load one section of a cache document if it exists and is still valid.

    Invalid, stale or unreadable cache files are treated as cache misses.
    """
    try:
        fingerprint = {
            **build_model_file_fingerprint(model_path),
            "path": _relative_path(cache_dir, model_path),
        }
        path_for = get_digest_cache_path if section == _DIGEST_SECTION else get_metadata_cache_path
        cache_path = path_for(cache_dir, model_path)
        with cache_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        if not _is_valid_cache_document(document, fingerprint, schema_version):
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
) -> None:
    """
    Save one cache document using an atomic replace.

    Cache write failures are intentionally non-fatal for the application.
    """
    try:
        path_for = get_digest_cache_path if section == _DIGEST_SECTION else get_metadata_cache_path
        cache_path = path_for(cache_dir, model_path)

        if fingerprint is None:
            fingerprint = build_model_file_fingerprint(model_path)

        # Preserve the captured size/mtime and path: delayed writes must not
        # label an old scan with the current file's identity.
        fingerprint = {
            **fingerprint,
            "path": Path(fingerprint["path"]).relative_to(cache_dir.parent.parent.resolve()).as_posix(),
        }
        cache_dir.mkdir(parents=True, exist_ok=True)

        document = {
            "schema_version": schema_version,
            "created_at": _utc_now_iso(),
            "model": {
                "name": model_name,
                **fingerprint,
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
) -> Optional[Dict[str, Any]]:
    """Load cached GGUF metadata if the cache file exists and is still valid."""
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    return _load_cache_document(
        cache_dir=cache_dir,
        model_path=model_path,
        schema_version=SCHEMA_VERSION,
        section=_METADATA_SECTION,
    )


def save_metadata_cache(
    cache_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    metadata: Dict[str, Any],
    *,
    fingerprint: Optional[Dict[str, Any]] = None,
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
    )


def load_digest_cache(
    cache_dir: str | Path,
    model_path: str | Path,
) -> Optional[Dict[str, Any]]:
    """Load the cached content digest if the cache file exists and is still valid."""
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    return _load_cache_document(
        cache_dir=cache_dir,
        model_path=model_path,
        schema_version=DIGEST_SCHEMA_VERSION,
        section=_DIGEST_SECTION,
    )


def save_digest_cache(
    cache_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    digest: Dict[str, Any],
    *,
    fingerprint: Optional[Dict[str, Any]] = None,
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
    )
