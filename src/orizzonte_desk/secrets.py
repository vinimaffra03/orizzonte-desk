from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import secrets
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from eth_account import Account

from orizzonte_desk.models import Environment, MainnetAuthorization


class SecretStoreError(RuntimeError):
    pass


if os.name == "nt":

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    CRYPTPROTECT_UI_FORBIDDEN = 0x01


def _blob(data: bytes) -> tuple[DATA_BLOB, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data)
    return DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _windows_dll() -> Any:
    if os.name != "nt":
        raise SecretStoreError("DPAPI está disponível somente no Windows")
    return cast(Any, vars(ctypes)["windll"])


def _last_error() -> int:
    return int(cast(Any, vars(ctypes)["GetLastError"])())


def protect(data: bytes, *, entropy: bytes = b"orizzonte-desk-v1") -> bytes:
    if os.name != "nt":
        raise SecretStoreError("DPAPI está disponível somente no Windows")
    input_blob, input_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(entropy)
    output_blob = DATA_BLOB()
    dll = _windows_dll()
    result = dll.crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "Orizzonte Desk",
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    _ = input_buffer, entropy_buffer
    if not result:
        raise SecretStoreError(f"CryptProtectData falhou: {_last_error()}")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        dll.kernel32.LocalFree(output_blob.pbData)


def unprotect(data: bytes, *, entropy: bytes = b"orizzonte-desk-v1") -> bytes:
    if os.name != "nt":
        raise SecretStoreError("DPAPI está disponível somente no Windows")
    input_blob, input_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(entropy)
    output_blob = DATA_BLOB()
    dll = _windows_dll()
    result = dll.crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    _ = input_buffer, entropy_buffer
    if not result:
        raise SecretStoreError(f"CryptUnprotectData falhou: {_last_error()}")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        dll.kernel32.LocalFree(output_blob.pbData)


class DPAPISecretStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, payload: dict[str, Any]) -> None:
        if not payload.get("secret_key") or not payload.get("account_address"):
            raise SecretStoreError("secret_key e account_address são obrigatórios")
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        encrypted = protect(encoded)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(base64.b64encode(encrypted))

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise SecretStoreError(f"Cofre não encontrado: {self.path}")
        encrypted = base64.b64decode(self.path.read_bytes(), validate=True)
        payload = json.loads(unprotect(encrypted).decode("utf-8"))
        if not isinstance(payload, dict):
            raise SecretStoreError("Conteúdo inválido no cofre DPAPI")
        return cast(dict[str, Any], payload)

    def exists(self) -> bool:
        return self.path.exists()

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()


class EnvironmentSecretManager:
    """Environment-separated API-wallet vaults protected by the current Windows user DPAPI."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, environment: Environment) -> Path:
        if environment not in {Environment.TESTNET, Environment.MAINNET}:
            raise SecretStoreError("Paper não possui API wallet")
        return self.directory / f"hyperliquid-{environment.value}.dpapi"

    def generate(
        self,
        environment: Environment,
        *,
        secret_key: str | None = None,
        account_address: str,
    ) -> dict[str, Any]:
        if self.path_for(environment).exists():
            raise SecretStoreError("Cofre já existe; use rotate")
        return self._save_new(
            environment,
            secret_key=secret_key,
            account_address=account_address,
            revoked_fingerprints=[],
        )

    def _save_new(
        self,
        environment: Environment,
        *,
        secret_key: str | None,
        account_address: str,
        revoked_fingerprints: list[str],
    ) -> dict[str, Any]:
        secret_key = secret_key or Account.create().key.hex()
        wallet_address = Account.from_key(secret_key).address.lower()
        account = account_address.lower()
        if not account.startswith("0x") or len(account) != 42:
            raise SecretStoreError("Endereço da conta principal inválido")
        if wallet_address == account:
            raise SecretStoreError("A API wallet deve ser diferente da conta principal")
        fingerprint = hashlib.sha256(wallet_address.encode()).hexdigest()[:16]
        if fingerprint in revoked_fingerprints:
            raise SecretStoreError("API wallet rotacionada não pode ser reutilizada")
        payload = {
            "schema_version": 1,
            "environment": environment.value,
            "secret_key": secret_key,
            "account_address": account,
            "wallet_address": wallet_address,
            "fingerprint": fingerprint,
            "revoked_fingerprints": sorted(set(revoked_fingerprints)),
            "created_at": datetime.now(UTC).isoformat(),
        }
        _write_encrypted(self.path_for(environment), payload, _environment_entropy(environment))
        return self.status(environment, verify=True)

    def load(self, environment: Environment) -> dict[str, Any]:
        payload = _read_encrypted(self.path_for(environment), _environment_entropy(environment))
        if payload.get("environment") != environment.value:
            raise SecretStoreError("Cofre pertence a outro ambiente")
        return payload

    def verify(self, environment: Environment) -> dict[str, Any]:
        payload = self.load(environment)
        derived = Account.from_key(str(payload["secret_key"])).address.lower()
        if derived != str(payload.get("wallet_address", "")).lower():
            raise SecretStoreError("API wallet diverge da chave armazenada")
        if derived == str(payload.get("account_address", "")).lower():
            raise SecretStoreError("API wallet não é exclusiva")
        return self.status(environment, verify=False) | {"verified": True}

    def rotate(
        self,
        environment: Environment,
        *,
        secret_key: str | None = None,
        account_address: str,
    ) -> dict[str, Any]:
        previous_payload = self.load(environment)
        before = self.status(environment, verify=False)
        revoked = [str(item) for item in previous_payload.get("revoked_fingerprints", [])]
        revoked.append(str(previous_payload["fingerprint"]))
        after = self._save_new(
            environment,
            secret_key=secret_key,
            account_address=account_address,
            revoked_fingerprints=revoked,
        )
        return {"rotated": True, "previous": before, "current": after}

    def status(self, environment: Environment, *, verify: bool = False) -> dict[str, Any]:
        path = self.path_for(environment)
        if not path.is_file():
            return {"environment": environment.value, "configured": False, "verified": False}
        if verify:
            return self.verify(environment)
        payload = self.load(environment)
        return {
            "environment": environment.value,
            "configured": True,
            "verified": False,
            "account_address": payload.get("account_address"),
            "wallet_address": payload.get("wallet_address"),
            "fingerprint": payload.get("fingerprint"),
            "created_at": payload.get("created_at"),
        }


class DPAPICapabilityStore:
    """One file per short-lived capability; plaintext tokens are never returned or persisted."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory / "capabilities"

    def issue(self, authorization: MainnetAuthorization) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        payload = authorization.model_dump(mode="json") | {
            "token": token,
            "token_hash": token_hash,
        }
        _write_encrypted(self._path(authorization.authorization_id), payload, self._entropy())
        return token_hash

    def load(self, authorization_id: str) -> dict[str, Any]:
        payload = _read_encrypted(self._path(authorization_id), self._entropy())
        if payload.get("authorization_id") != authorization_id:
            raise SecretStoreError("Capability DPAPI possui id divergente")
        return payload

    def status(self, authorization_id: str) -> dict[str, Any]:
        path = self._path(authorization_id)
        if not path.is_file():
            return {"authorization_id": authorization_id, "available": False}
        payload = self.load(authorization_id)
        return {
            "authorization_id": authorization_id,
            "available": True,
            "release_id": payload.get("release_id"),
            "certificate_id": payload.get("certificate_id"),
            "budget_usdc": payload.get("budget_usdc"),
            "expires_at": payload.get("expires_at"),
        }

    def delete(self, authorization_id: str) -> None:
        path = self._path(authorization_id)
        if path.exists():
            path.unlink()

    def _path(self, authorization_id: str) -> Path:
        if not authorization_id or any(
            char not in "0123456789abcdef-" for char in authorization_id
        ):
            raise SecretStoreError("Id de capability inválido")
        return self.directory / f"{authorization_id}.dpapi"

    @staticmethod
    def _entropy() -> bytes:
        return b"orizzonte-desk-mainnet-capability-v1"


def _environment_entropy(environment: Environment) -> bytes:
    return f"orizzonte-desk-wallet-{environment.value}-v1".encode()


def _write_encrypted(path: Path, payload: dict[str, Any], entropy: bytes) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    encrypted = base64.b64encode(protect(encoded, entropy=entropy))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encrypted)
    temporary.replace(path)


def _read_encrypted(path: Path, entropy: bytes) -> dict[str, Any]:
    if not path.is_file():
        raise SecretStoreError(f"Cofre não encontrado: {path}")
    try:
        encrypted = base64.b64decode(path.read_bytes(), validate=True)
        payload = json.loads(unprotect(encrypted, entropy=entropy).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SecretStoreError("Cofre DPAPI inválido ou corrompido") from exc
    if not isinstance(payload, dict):
        raise SecretStoreError("Conteúdo inválido no cofre DPAPI")
    return cast(dict[str, Any], payload)
