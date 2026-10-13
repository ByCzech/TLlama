"""Read the installed TLlama version without loading inference code."""

from importlib import metadata


def get_version() -> str:
    try:
        return metadata.version("tllama")
    except metadata.PackageNotFoundError:
        return "0.0.0-dev"
