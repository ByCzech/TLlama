from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _cache_key(model_path: Path) -> str:
    """Build a stable cache key from an absolute model path."""
    resolved = str(model_path.resolve())
    return sha256(resolved.encode("utf-8")).hexdigest()


def get_metadata_cache_path(cache_dir: Path, model_path: str | Path) -> Path:
    """Return the JSON cache file path for a model file."""
    path = Path(model_path)
    return cache_dir / f"{_cache_key(path)}.json"


def build_model_file_fingerprint(model_path: str | Path) -> Dict[str, Any]:
    """Build a lightweight model file fingerprint for cache invalidation."""
    path = Path(model_path)
    stats = path.stat()

    return {
        "path": str(path.resolve()),
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _is_valid_cache_document(document: Any, fingerprint: Dict[str, Any]) -> bool:
    """Check whether a metadata cache document matches the current model file."""
    if not isinstance(document, dict):
        return False

    if document.get("schema_version") != SCHEMA_VERSION:
        return False

    model = document.get("model")
    if not isinstance(model, dict):
        return False

    return (
        model.get("path") == fingerprint["path"]
        and model.get("size") == fingerprint["size"]
        and model.get("mtime_ns") == fingerprint["mtime_ns"]
    )


def load_metadata_cache(
    cache_dir: str | Path,
    model_path: str | Path,
) -> Optional[Dict[str, Any]]:
    """
    Load cached model metadata if the cache file exists and is still valid.

    Invalid, stale or unreadable cache files are treated as cache misses.
    """
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    try:
        fingerprint = build_model_file_fingerprint(model_path)
        cache_path = get_metadata_cache_path(cache_dir, model_path)

        with cache_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        if not _is_valid_cache_document(document, fingerprint):
            return None

        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            return None

        return metadata
    except FileNotFoundError:
        return None
    except Exception:
        return None


def save_metadata_cache(
    cache_dir: str | Path,
    model_name: str,
    model_path: str | Path,
    metadata: Dict[str, Any],
) -> None:
    """
    Save model metadata to a JSON cache file using an atomic replace.

    Cache write failures are intentionally non-fatal for the application.
    """
    cache_dir = Path(cache_dir)
    model_path = Path(model_path)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        fingerprint = build_model_file_fingerprint(model_path)
        cache_path = get_metadata_cache_path(cache_dir, model_path)

        document = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now_iso(),
            "model": {
                "name": model_name,
                **fingerprint,
            },
            "metadata": metadata,
        }

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=str(cache_dir),
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


def delete_metadata_cache(
    cache_dir: str | Path,
    model_path: str | Path,
) -> None:
    """Delete a model metadata cache file if it exists."""
    try:
        cache_path = get_metadata_cache_path(Path(cache_dir), model_path)
        cache_path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return
