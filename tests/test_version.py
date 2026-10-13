"""The API version identifies TLlama, including an uninstalled checkout."""

from importlib import metadata

import pytest

from tllama.version import get_version


def test_api_returns_distribution_version(client, monkeypatch):
    requested = []

    def installed_version(name):
        requested.append(name)
        return "0.6.0.dev1+build11552"

    monkeypatch.setattr(metadata, "version", installed_version)
    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": "0.6.0.dev1+build11552"}
    assert requested == ["tllama"]


def test_api_works_without_distribution_metadata(client, monkeypatch):
    def missing_version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing_version)
    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": "0.0.0-dev"}


def test_real_installed_metadata_is_the_version_source(client):
    try:
        installed = metadata.version("tllama")
    except metadata.PackageNotFoundError:
        pytest.skip("TLlama distribution is not installed")

    assert get_version() == installed
    assert client.get("/api/version").json() == {"version": installed}
