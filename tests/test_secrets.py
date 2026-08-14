from __future__ import annotations

import pytest

import orizzonte_desk.secrets as secret_module
from orizzonte_desk.secrets import DPAPISecretStore, SecretStoreError


def test_secret_store_roundtrip_without_exposing_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(secret_module, "protect", lambda value: value[::-1])
    monkeypatch.setattr(secret_module, "unprotect", lambda value: value[::-1])
    path = tmp_path / "wallet.dpapi"
    store = DPAPISecretStore(path)
    payload = {"account_address": "0x" + "1" * 40, "secret_key": "0x" + "2" * 64}

    store.save(payload)
    assert store.exists()
    assert payload["secret_key"].encode() not in path.read_bytes()
    assert store.load() == payload
    store.delete()
    assert not store.exists()


def test_secret_store_fails_closed_for_missing_fields_and_file(tmp_path) -> None:
    store = DPAPISecretStore(tmp_path / "missing.dpapi")
    with pytest.raises(SecretStoreError, match="obrigatórios"):
        store.save({"account_address": "0x" + "1" * 40})
    with pytest.raises(SecretStoreError, match="não encontrado"):
        store.load()


def test_dpapi_is_explicitly_windows_only(monkeypatch) -> None:
    monkeypatch.setattr(secret_module.os, "name", "posix")
    with pytest.raises(SecretStoreError, match="somente no Windows"):
        secret_module.protect(b"secret")
    with pytest.raises(SecretStoreError, match="somente no Windows"):
        secret_module.unprotect(b"secret")
