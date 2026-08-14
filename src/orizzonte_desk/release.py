from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from orizzonte_desk.config import Settings
from orizzonte_desk.data import stable_fingerprint
from orizzonte_desk.ml import research_code_fingerprint
from orizzonte_desk.paths import AppPaths


class ReleaseError(RuntimeError):
    """Raised when a release cannot be proven safe and reproducible."""


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_id: str
    created_at: datetime
    git_commit: str
    artifacts: dict[str, dict[str, str]]
    approval_passed: bool
    approved: bool = False
    approved_at: datetime | None = None
    verified_at: datetime | None = None


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hashes(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "model_hash" and isinstance(item, str) and item:
                found.add(item)
            found.update(_model_hashes(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_model_hashes(item))
    return found


class ReleaseManager:
    """Builds a tamper-evident local release without enabling mainnet."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.directory = paths.reports / "releases"

    @property
    def approved_pointer(self) -> Path:
        return self.directory / "approved.json"

    def build(self) -> ReleaseManifest:
        commit, dirty = self._git_identity()
        if dirty:
            raise ReleaseError("A release exige worktree limpo e commitado")
        required = {
            "config": self.paths.config,
            "model": self.paths.models / "promoted.joblib",
            "model_pointer": self.paths.models / "promoted.json",
            "research_approval": self.paths.reports / "live-approval.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise ReleaseError(f"Artefatos obrigatórios ausentes: {', '.join(missing)}")

        pointer = json.loads(required["model_pointer"].read_text(encoding="utf-8"))
        approval = json.loads(required["research_approval"].read_text(encoding="utf-8"))
        if not approval.get("passed", False):
            raise ReleaseError("Gate combinado de pesquisa está reprovado")
        actual_model_hash = sha256_path(required["model"])
        pointer_hash = str(pointer.get("promoted_hash") or pointer.get("model_hash") or "")
        if pointer_hash != actual_model_hash:
            raise ReleaseError("Hash do modelo promovido diverge do ponteiro")
        approved_hashes = _model_hashes(approval)
        if actual_model_hash not in approved_hashes:
            raise ReleaseError("Gate combinado não está vinculado ao modelo promovido")
        binding = approval.get("release_binding")
        pointer_binding = pointer.get("release_binding")
        if not isinstance(binding, dict) or not isinstance(pointer_binding, dict):
            raise ReleaseError("Release binding auditável está ausente")
        expected_config = stable_fingerprint(
            Settings.load(required["config"]).model_dump(mode="json")
        )
        if (
            binding.get("evaluation_scope") != "combined_release"
            or binding.get("model_hash") != actual_model_hash
            or approval.get("model_hash") != actual_model_hash
            or binding.get("commit_hash") != commit
            or binding.get("config_fingerprint") != expected_config
            or binding.get("code_hash") != research_code_fingerprint()
        ):
            raise ReleaseError("Gate combinado diverge do modelo, configuração, código ou commit")
        for key in ("config_fingerprint", "code_hash", "commit_hash"):
            if pointer_binding.get(key) != binding.get(key):
                raise ReleaseError(f"Ponteiro do modelo diverge do gate combinado: {key}")
        evaluated_datasets = set(binding.get("dataset_hashes", []))
        training_datasets = set(pointer_binding.get("dataset_hashes", []))
        if (
            len(evaluated_datasets) < 2
            or not training_datasets
            or not training_datasets <= evaluated_datasets
            or not binding.get("protocol_hashes")
        ):
            raise ReleaseError("Datasets de treino, holdout e protocolo não estão vinculados")

        artifacts = {
            name: {"path": str(path.resolve()), "sha256": sha256_path(path)}
            for name, path in required.items()
        }
        fingerprint = json.dumps(
            {"git_commit": commit, "artifacts": artifacts}, sort_keys=True
        ).encode("utf-8")
        release_id = f"release-{hashlib.sha256(fingerprint).hexdigest()[:16]}"
        manifest = ReleaseManifest(
            release_id=release_id,
            created_at=datetime.now(UTC),
            git_commit=commit,
            artifacts=artifacts,
            approval_passed=True,
        )
        destination = self.directory / release_id
        destination.mkdir(parents=True, exist_ok=True)
        self._write(destination / "manifest.json", manifest)
        return manifest

    def verify(self, release_id: str) -> ReleaseManifest:
        manifest_path = self.directory / release_id / "manifest.json"
        if not manifest_path.is_file():
            raise ReleaseError(f"Release não encontrada: {release_id}")
        manifest = ReleaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest)
        verified = manifest.model_copy(update={"verified_at": datetime.now(UTC)})
        self._write(manifest_path, verified)
        if verified.approved:
            self._write_approved_pointer(verified, manifest_path)
        return verified

    def approve(self, release_id: str, confirmation: str) -> ReleaseManifest:
        expected = f"APPROVE RELEASE {release_id}"
        if confirmation != expected:
            raise ReleaseError(f"Confirmação inválida. Digite exatamente: {expected}")
        verified = self.verify(release_id)
        approved = verified.model_copy(update={"approved": True, "approved_at": datetime.now(UTC)})
        manifest_path = self.directory / release_id / "manifest.json"
        self._write(manifest_path, approved)
        self._write_approved_pointer(approved, manifest_path)
        return approved

    def approved(self) -> ReleaseManifest | None:
        if not self.approved_pointer.is_file():
            return None
        pointer = json.loads(self.approved_pointer.read_text(encoding="utf-8"))
        manifest_path = Path(pointer["manifest"])
        if not manifest_path.is_file() or sha256_path(manifest_path) != pointer["manifest_sha256"]:
            raise ReleaseError("Ponteiro de release aprovada foi alterado")
        manifest = ReleaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if not manifest.approved:
            raise ReleaseError("Ponteiro referencia release não aprovada")
        self._validate_manifest(manifest)
        return manifest

    def _validate_manifest(self, manifest: ReleaseManifest) -> None:
        commit, dirty = self._git_identity()
        if dirty or commit != manifest.git_commit:
            raise ReleaseError("Commit atual não corresponde à release ou worktree está sujo")
        for name, artifact in manifest.artifacts.items():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_path(path) != artifact["sha256"]:
                raise ReleaseError(f"Artefato inválido ou alterado: {name}")

    def _write_approved_pointer(self, manifest: ReleaseManifest, path: Path) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.approved_pointer.write_text(
            json.dumps(
                {
                    "release_id": manifest.release_id,
                    "manifest": str(path.resolve()),
                    "manifest_sha256": sha256_path(path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _git_identity(self) -> tuple[str, bool]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.paths.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.paths.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseError("Não foi possível identificar o commit Git") from exc
        return commit, bool(status)

    @staticmethod
    def _write(path: Path, manifest: ReleaseManifest) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
