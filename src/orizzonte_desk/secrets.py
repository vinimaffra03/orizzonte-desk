from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any, cast


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
