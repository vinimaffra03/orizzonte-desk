from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from orizzonte_desk.chaos import (
    ChaosValidationError,
    TestnetChaosContext,
    TestnetChaosRunner,
)
from orizzonte_desk.config import Settings
from orizzonte_desk.constants import LIVE_CONFIRMATION_PREFIX
from orizzonte_desk.exchange import (
    AccountSnapshot,
    HyperliquidGateway,
    PaperGateway,
    TradingGateway,
)
from orizzonte_desk.models import (
    AgentState,
    AgentStatus,
    Environment,
    MainnetAuthorization,
    Side,
    Signal,
    TestnetCertificate,
)
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.release import ReleaseError, ReleaseManager, ReleaseManifest
from orizzonte_desk.secrets import (
    DPAPICapabilityStore,
    EnvironmentSecretManager,
    SecretStoreError,
)
from orizzonte_desk.storage import StateStore


class ControlError(RuntimeError):
    pass


class AgentController:
    def __init__(
        self,
        paths: AppPaths,
        settings: Settings,
        store: StateStore,
        *,
        chaos_runner: TestnetChaosRunner | None = None,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.store = store
        self.wallets = EnvironmentSecretManager(paths.secrets)
        self.capabilities = DPAPICapabilityStore(paths.secrets)
        self.chaos_runner = chaos_runner or TestnetChaosRunner()
        self._paper_gateway = PaperGateway(
            settings.backtest.initial_capital,
            store=store,
            taker_fee=settings.execution.taker_fee,
        )

    def expected_confirmation(self, environment: Environment, budget_usdc: float) -> str:
        return f"{LIVE_CONFIRMATION_PREFIX} {environment.value.upper()} {budget_usdc:.2f}"

    def arm(
        self,
        *,
        environment: Environment,
        budget_usdc: float,
        confirmation: str,
    ) -> AgentState:
        state = self.store.agent_state()
        if state.status is not AgentStatus.DISARMED:
            raise ControlError(f"Agente precisa estar desarmado; estado atual: {state.status}")
        if budget_usdc <= 0:
            raise ControlError("Orçamento deve ser positivo")
        if (
            environment is Environment.MAINNET
            and budget_usdc > self.settings.mainnet.initial_budget_cap_usdc
        ):
            raise ControlError(
                f"Budget mainnet excede cap inicial de "
                f"{self.settings.mainnet.initial_budget_cap_usdc:.2f} USDC"
            )
        expected = self.expected_confirmation(environment, budget_usdc)
        if confirmation != expected:
            raise ControlError(f"Confirmação inválida. Digite exatamente: {expected}")
        self.paths.assert_free_space(self.settings.app.minimum_free_gb)
        release: ReleaseManifest | None = None
        try:
            gateway = self.gateway(environment)
        except SecretStoreError as exc:
            label = "Mainnet" if environment is Environment.MAINNET else environment.value
            raise ControlError(f"{label}: cofre ausente ou inválido: {exc}") from exc
        if environment is not Environment.PAPER:
            self._validate_research_approval()
            self._validate_promoted_model()
            release = self._validate_approved_release()
            self._assert_no_persistent_lock()
        preflight: dict[str, object] = {}
        if isinstance(gateway, HyperliquidGateway):
            try:
                preflight = gateway.preflight(require_empty=True)
            except Exception as exc:
                raise ControlError(f"Preflight da Hyperliquid falhou: {exc}") from exc
        snapshot = gateway.reconcile()
        if snapshot.open_orders or snapshot.positions:
            raise ControlError(
                "Existem ordens ou posições preexistentes. O agente não misturará atividade manual."
            )
        if budget_usdc > snapshot.equity:
            raise ControlError(
                f"Orçamento {budget_usdc:.2f} excede equity disponível {snapshot.equity:.2f}"
            )
        account = None
        wallet = None
        if environment is not Environment.PAPER:
            wallet_payload = self.wallets.load(environment)
            account = str(wallet_payload["account_address"]).lower()
            wallet = str(wallet_payload["wallet_address"]).lower()
        authorization: MainnetAuthorization | None = None
        session_id = uuid.uuid4().hex
        if environment is Environment.MAINNET:
            if release is None or account is None or wallet is None:
                raise ControlError("Bindings mainnet incompletos")
            authorization = self._consume_mainnet_capability(
                release=release,
                budget_usdc=budget_usdc,
                account_address=account,
                wallet_address=wallet,
                session_id=session_id,
            )
        armed = AgentState(
            status=AgentStatus.ARMED,
            environment=environment,
            budget_usdc=budget_usdc,
            account_address=account,
            armed_at=datetime.now(UTC),
            last_heartbeat=datetime.now(UTC),
            metadata={
                "preflight_equity": snapshot.equity,
                "isolated_leverage": 10,
                "preflight": preflight,
                "release_id": (release.release_id if release is not None else None),
                "reconciled_at": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "authorization_id": (
                    authorization.authorization_id if authorization is not None else None
                ),
                "certificate_id": (
                    authorization.certificate_id if authorization is not None else None
                ),
            },
        )
        self.store.save_agent_state(armed)
        self.store.event(
            "control",
            f"Agente armado em {environment.value}",
            payload={"budget_usdc": budget_usdc},
        )
        return armed

    def start(self) -> AgentState:
        state = self.store.agent_state()
        if state.status not in {AgentStatus.ARMED, AgentStatus.PAUSED}:
            raise ControlError(f"Não é possível iniciar a partir de {state.status}")
        if state.environment is not Environment.PAPER:
            self._revalidate_session(state)
        running = state.model_copy(
            update={"status": AgentStatus.RUNNING, "last_heartbeat": datetime.now(UTC)}
        )
        self.store.save_agent_state(running)
        self.store.event("control", "Agente iniciado")
        return running

    def _revalidate_session(self, state: AgentState) -> None:
        self._validate_research_approval()
        self._validate_promoted_model()
        release = self._validate_approved_release()
        if state.metadata.get("release_id") != release.release_id:
            raise ControlError("Release da sessão mudou; desarme e arme novamente")
        self._assert_no_persistent_lock()
        account = state.account_address or "paper"
        known_symbols = {
            item["symbol"]
            for item in self.store.positions(
                environment=state.environment,
                account_address=account,
            )
        }
        known_cloids = {
            item["client_order_id"]
            for item in self.store.orders(
                environment=state.environment,
                account_address=account,
            )
        }
        gateway = self.gateway(state.environment)
        if not isinstance(gateway, HyperliquidGateway):
            raise ControlError("Gateway live inválido")
        try:
            preflight = gateway.preflight(require_empty=state.status is AgentStatus.ARMED)
            snapshot = gateway.reconcile()
        except Exception as exc:
            raise ControlError(f"Revalidação da sessão falhou: {exc}") from exc
        manual_symbols = {
            str(item.get("coin"))
            for item in snapshot.positions
            if item.get("coin") not in known_symbols
        }
        manual_orders = {
            str(item.get("cloid") or f"oid:{item.get('oid')}")
            for item in snapshot.open_orders
            if str(item.get("cloid") or f"oid:{item.get('oid')}") not in known_cloids
        }
        if manual_symbols or manual_orders:
            raise ControlError(
                "Exposição manual detectada durante resume: "
                f"positions={sorted(manual_symbols)}, orders={sorted(manual_orders)}"
            )
        if state.environment is Environment.MAINNET and state.metadata.get(
            "requires_mainnet_reauthorization", False
        ):
            wallet_payload = self.wallets.load(Environment.MAINNET)
            resumed_session_id = uuid.uuid4().hex
            authorization = self._consume_mainnet_capability(
                release=release,
                budget_usdc=float(state.budget_usdc or 0),
                account_address=str(wallet_payload["account_address"]),
                wallet_address=str(wallet_payload["wallet_address"]),
                session_id=resumed_session_id,
            )
            state.metadata["authorization_id"] = authorization.authorization_id
            state.metadata["session_id"] = resumed_session_id
            state.metadata["certificate_id"] = authorization.certificate_id
            state.metadata.pop("requires_mainnet_reauthorization", None)
        state.metadata["preflight"] = preflight
        state.metadata["reconciled_at"] = datetime.now(UTC).isoformat()
        state.metadata.pop("restart_reconcile_required", None)
        self.store.save_agent_state(state)

    def pause(self) -> AgentState:
        state = self.store.agent_state()
        if state.status is not AgentStatus.RUNNING:
            raise ControlError("Somente um agente em execução pode ser pausado")
        metadata = dict(state.metadata)
        if state.environment is Environment.MAINNET:
            metadata["requires_mainnet_reauthorization"] = True
        paused = state.model_copy(update={"status": AgentStatus.PAUSED, "metadata": metadata})
        self.store.save_agent_state(paused)
        self.store.event("control", "Novas entradas pausadas")
        return paused

    def flatten(self) -> AgentState:
        state = self.store.agent_state()
        if state.status is AgentStatus.DISARMED:
            raise ControlError("Agente já está desarmado")
        gateway = self.gateway(state.environment)
        responses = gateway.flatten_all(slippage=0.02)
        locked = state.model_copy(update={"status": AgentStatus.LOCKED})
        self.store.save_agent_state(locked)
        self.store.event(
            "risk",
            "Flatten solicitado pelo operador; agente bloqueado",
            level="WARNING",
            payload={"responses": responses},
        )
        return locked

    def disarm(self) -> AgentState:
        state = self.store.agent_state()
        if state.status is AgentStatus.RUNNING:
            raise ControlError("Pause ou flatten antes de desarmar")
        gateway = self.gateway(state.environment)
        snapshot = gateway.snapshot()
        if snapshot.positions:
            raise ControlError("Há posições abertas; use flatten antes de desarmar")
        gateway.cancel_all()
        disarmed = AgentState(environment=Environment.PAPER)
        self.store.save_agent_state(disarmed)
        self.store.event("control", "Agente desarmado")
        return disarmed

    def gateway(self, environment: Environment) -> TradingGateway:
        if environment is Environment.PAPER:
            return self._paper_gateway
        payload = self.wallets.load(environment)
        return HyperliquidGateway(
            secret_key=str(payload["secret_key"]),
            account_address=str(payload["account_address"]),
            environment=environment,
            store=self.store,
        )

    def _validate_research_approval(self) -> None:
        approval = self.paths.reports / "live-approval.json"
        if not approval.exists():
            raise ControlError("Gate combinado ausente: execute `orizzonte backtest compare`")
        payload = json.loads(approval.read_text(encoding="utf-8"))
        if not payload.get("passed", False):
            raise ControlError("Gate combinado de pesquisa está reprovado")
        if not _strings_for_key(payload, "dataset_hashes"):
            raise ControlError("Gate combinado não identifica os datasets avaliados")

    def _validate_promoted_model(self) -> None:
        promoted = self.paths.models / "promoted.json"
        model = self.paths.models / "promoted.joblib"
        if not promoted.exists() or not model.exists():
            raise ControlError("Nenhum modelo promovido está disponível")
        pointer = json.loads(promoted.read_text(encoding="utf-8"))
        expected = str(pointer.get("promoted_hash") or pointer.get("model_hash") or "")
        actual = _sha256(model)
        if not expected or expected != actual:
            raise ControlError("Hash do modelo promovido diverge do ponteiro")
        approval = json.loads(
            (self.paths.reports / "live-approval.json").read_text(encoding="utf-8")
        )
        if actual not in _strings_for_key(approval, "model_hash"):
            raise ControlError("Gate combinado não está vinculado ao modelo promovido")

    def _validate_approved_release(self) -> ReleaseManifest:
        try:
            approved = ReleaseManager(self.paths).approved()
        except ReleaseError as exc:
            raise ControlError(f"Release aprovada inválida: {exc}") from exc
        if approved is None:
            raise ControlError("Release verificada e aprovada está ausente")
        return approved

    def _assert_no_persistent_lock(self) -> None:
        for name in ("daily_loss", "drawdown", "unprotected_position", "connectivity"):
            lock = self.store.lock(name)
            if name == "daily_loss" and lock:
                risk_day = lock.get("payload", {}).get("risk_day")
                if risk_day and risk_day != datetime.now(UTC).strftime("%Y-%m-%d"):
                    self.store.clear_lock(name)
                    continue
            if lock and lock.get("locked"):
                raise ControlError(f"Lock persistente ativo ({name}): {lock.get('reason')}")

    def testnet_preflight(self) -> dict[str, object]:
        """Run all read-only live checks without arming or submitting an order."""
        self._assert_testnet_operation_allowed()
        self._validate_research_approval()
        self._validate_promoted_model()
        release = self._validate_approved_release()
        self._assert_no_persistent_lock()
        gateway = self.gateway(Environment.TESTNET)
        if not isinstance(gateway, HyperliquidGateway):
            raise ControlError("Gateway testnet indisponível")
        try:
            result = gateway.preflight(require_empty=True)
            gateway.reconcile()
        except Exception as exc:
            raise ControlError(f"Preflight testnet falhou: {exc}") from exc
        return result | {"release_id": release.release_id}

    def reconcile(self) -> dict[str, object]:
        state = self.store.agent_state()
        if state.environment is Environment.MAINNET and state.status is AgentStatus.DISARMED:
            raise ControlError("Reconciliação mainnet requer sessão explicitamente armada")
        snapshot = self.gateway(state.environment).reconcile()
        return {
            "environment": state.environment.value,
            "equity": snapshot.equity,
            "positions": len(snapshot.positions),
            "orders": len(snapshot.open_orders),
        }

    def testnet_reconcile(self) -> dict[str, object]:
        self._assert_testnet_operation_allowed()
        gateway = self.gateway(Environment.TESTNET)
        snapshot = gateway.reconcile()
        payload = self.wallets.load(Environment.TESTNET)
        return {
            "environment": Environment.TESTNET.value,
            "equity": snapshot.equity,
            "positions": len(snapshot.positions),
            "orders": len(snapshot.open_orders),
            "fills_persisted": len(
                self.store.fills(
                    environment=Environment.TESTNET,
                    account_address=str(payload["account_address"]),
                )
            ),
        }

    def testnet_smoke(self, *, budget_usdc: float, confirmation: str) -> dict[str, object]:
        """Explicit, bounded testnet-only lifecycle; it cannot select mainnet."""
        expected = f"TESTNET SMOKE {budget_usdc:.2f}"
        if confirmation != expected:
            raise ControlError(f"Confirmação inválida. Digite exatamente: {expected}")
        if budget_usdc < 20:
            raise ControlError("Smoke testnet exige budget mock mínimo de 20 USDC")
        if self.store.agent_state().status is not AgentStatus.DISARMED:
            raise ControlError("Smoke testnet exige agente desarmado")
        preflight = self.testnet_preflight()
        gateway = self.gateway(Environment.TESTNET)
        if not isinstance(gateway, HyperliquidGateway):
            raise ControlError("Smoke recusado: gateway não é testnet")
        if gateway.environment is not Environment.TESTNET:
            raise ControlError("Smoke jamais pode executar em mainnet")
        snapshot = gateway.snapshot()
        if budget_usdc > snapshot.equity:
            raise ControlError("Budget mock excede a equity testnet")
        price = snapshot.mids.get("BTC")
        if not price or price <= 0:
            raise ControlError("Mid de BTC indisponível no testnet")
        asset = gateway.asset_metadata("BTC")
        size = max(
            asset.size_increment,
            math.ceil((12 / price) / asset.size_increment) * asset.size_increment,
        )
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC",
            side=Side.LONG,
            score=1,
            probability=1,
            entry_reference=price,
            stop_distance=price * 0.01,
            atr=price * 0.005,
            regime="bull",
            reasons=("testnet_smoke",),
        )
        lifecycle: dict[str, object] = {"preflight": preflight}
        try:
            lifecycle["dead_man"] = gateway.schedule_dead_man(30)
            lifecycle["entry"] = gateway.place_entry_with_protection(
                signal,
                size=size,
                stop_price=price * 0.99,
                take_profit_price=price * 1.01,
                slippage=0.01,
            )
            after_entry = gateway.reconcile()
            lifecycle["after_entry"] = _snapshot_counts(after_entry)
            lifecycle["protection_evidence"] = {
                "position": next(
                    (item for item in after_entry.positions if item.get("coin") == "BTC"),
                    None,
                ),
                "orders": [item for item in after_entry.open_orders if item.get("coin") == "BTC"],
            }
            # Constructing a new adapter simulates a process restart and forces REST recovery.
            restarted = self.gateway(Environment.TESTNET)
            lifecycle["after_restart"] = _snapshot_counts(restarted.reconcile())
            lifecycle["flatten"] = restarted.flatten_all(slippage=0.02)
            lifecycle["final"] = _snapshot_counts(restarted.reconcile())
        except Exception as exc:
            try:
                gateway.flatten_all(slippage=0.03)
            except Exception as flatten_exc:
                self.store.latch_lock(
                    "unprotected_position",
                    reason="Smoke testnet falhou e flatten não foi confirmado",
                    payload={"error": str(exc), "flatten_error": str(flatten_exc)},
                )
            raise ControlError(f"Smoke testnet falhou fechado: {exc}") from exc
        final = lifecycle.get("final")
        if not isinstance(final, dict) or final.get("positions") or final.get("orders"):
            self.store.latch_lock(
                "unprotected_position",
                reason="Smoke testnet terminou com estado residual",
                payload={"final": final},
            )
            raise ControlError("Smoke testnet deixou ordens ou posições residuais")
        try:
            chaos_report = self.chaos_runner.run(TestnetChaosContext(lifecycle=lifecycle))
        except ChaosValidationError as exc:
            raise ControlError(f"Certificado testnet recusado pelo chaos gate: {exc}") from exc
        if not self.chaos_runner.verify(chaos_report):
            raise ControlError("Certificado testnet recusado: relatório de caos inválido")
        lifecycle["chaos"] = chaos_report
        self.store.event("testnet", "Smoke testnet concluído", payload=lifecycle)
        release = self._validate_approved_release()
        model_hash, gates_hash = _release_bindings(release)
        account_address = str(preflight.get("account_address", "")).lower()
        wallet_address = str(preflight.get("wallet_address", "")).lower()
        evidence = (
            _json_hash(lifecycle),
            _json_hash(
                self.store.fills(
                    environment=Environment.TESTNET,
                    account_address=account_address,
                )
            ),
        )
        certificate = TestnetCertificate.build(
            release_id=release.release_id,
            model_hash=model_hash,
            gates_hash=gates_hash,
            account_address=account_address,
            wallet_address=wallet_address,
            evidence_hashes=evidence,
            required_scenarios=tuple(cast(list[str], chaos_report["required_scenarios"])),
            scenario_results=dict(cast(dict[str, bool], chaos_report["results"])),
            scenario_hashes=dict(cast(dict[str, str], chaos_report["scenario_hashes"])),
            chaos_report_hash=str(chaos_report["report_hash"]),
        )
        if not certificate.verify_content_address():
            raise ControlError("Certificado testnet falhou na verificação content-addressed")
        self.store.save_testnet_certificate(certificate)
        lifecycle["certificate"] = certificate.model_dump(mode="json")
        return lifecycle

    def secret_generate(
        self,
        environment: Environment,
        *,
        secret_key: str | None = None,
        account_address: str,
    ) -> dict[str, Any]:
        self._assert_secret_mutation_allowed(environment)
        try:
            return self.wallets.generate(
                environment,
                secret_key=secret_key,
                account_address=account_address,
            )
        except Exception as exc:
            raise ControlError(f"Falha ao gerar cofre {environment.value}: {exc}") from exc

    def secret_verify(self, environment: Environment) -> dict[str, Any]:
        try:
            return self.wallets.verify(environment)
        except Exception as exc:
            raise ControlError(f"Falha ao verificar cofre {environment.value}: {exc}") from exc

    def secret_rotate(
        self,
        environment: Environment,
        *,
        secret_key: str | None = None,
        account_address: str,
    ) -> dict[str, Any]:
        self._assert_secret_mutation_allowed(environment)
        try:
            return self.wallets.rotate(
                environment,
                secret_key=secret_key,
                account_address=account_address,
            )
        except Exception as exc:
            raise ControlError(f"Falha ao rotacionar cofre {environment.value}: {exc}") from exc

    def secret_status(self, environment: Environment) -> dict[str, Any]:
        try:
            return self.wallets.status(environment, verify=False)
        except Exception as exc:
            raise ControlError(f"Falha ao consultar cofre {environment.value}: {exc}") from exc

    def issue_mainnet_authorization(
        self, *, budget_usdc: float, confirmation: str
    ) -> dict[str, Any]:
        if budget_usdc <= 0 or budget_usdc > self.settings.mainnet.initial_budget_cap_usdc:
            raise ControlError(
                f"Budget deve estar entre 0 e {self.settings.mainnet.initial_budget_cap_usdc:.2f}"
            )
        state = self.store.agent_state()
        if state.status is not AgentStatus.DISARMED and not (
            state.status is AgentStatus.PAUSED and state.environment is Environment.MAINNET
        ):
            raise ControlError("Autorização mainnet exige agente desarmado ou mainnet pausada")
        release = self._validate_approved_release()
        self._validate_research_approval()
        self._validate_promoted_model()
        self._assert_no_persistent_lock()
        certificate = self.store.latest_testnet_certificate()
        if certificate is None or not certificate.verify_content_address():
            raise ControlError("Certificado testnet content-addressed válido está ausente")
        bindings = _mainnet_release_bindings(release)
        model_hash = bindings["model_hash"]
        gates_hash = bindings["gates_hash"]
        if (
            certificate.release_id != release.release_id
            or certificate.model_hash != model_hash
            or certificate.gates_hash != gates_hash
        ):
            raise ControlError("Certificado testnet não corresponde à release/modelo/gates")
        try:
            wallet_status = self.wallets.verify(Environment.MAINNET)
            wallet_payload = self.wallets.load(Environment.MAINNET)
        except SecretStoreError as exc:
            raise ControlError(f"Cofre mainnet inválido: {exc}") from exc
        account = str(wallet_payload["account_address"]).lower()
        wallet = str(wallet_payload["wallet_address"]).lower()
        expected = f"AUTHORIZE MAINNET {release.release_id} {account} {budget_usdc:.2f}"
        if confirmation != expected:
            raise ControlError(f"Confirmação inválida. Digite exatamente: {expected}")
        if certificate.account_address != account:
            raise ControlError("Certificado testnet não vincula a mesma conta principal")
        gateway = self.gateway(Environment.MAINNET)
        if not isinstance(gateway, HyperliquidGateway):
            raise ControlError("Gateway mainnet inválido")
        try:
            gateway.preflight(require_empty=state.status is AgentStatus.DISARMED)
        except Exception as exc:
            raise ControlError(f"Preflight mainnet read-only falhou: {exc}") from exc
        issued_at = datetime.now(UTC)
        authorization = MainnetAuthorization(
            authorization_id=uuid.uuid4().hex,
            release_id=release.release_id,
            certificate_id=certificate.certificate_id,
            model_hash=model_hash,
            gates_hash=gates_hash,
            git_commit=bindings["git_commit"],
            config_sha256=bindings["config_sha256"],
            config_fingerprint=bindings["config_fingerprint"],
            code_hash=bindings["code_hash"],
            account_address=account,
            wallet_address=wallet,
            budget_usdc=budget_usdc,
            issued_at=issued_at,
            expires_at=issued_at
            + timedelta(seconds=self.settings.mainnet.authorization_ttl_seconds),
        )
        token_hash = self.capabilities.issue(authorization)
        try:
            self.store.save_mainnet_authorization(authorization, token_hash=token_hash)
        except Exception:
            self.capabilities.delete(authorization.authorization_id)
            raise
        self.store.set(
            "pending_mainnet_authorization",
            {"authorization_id": authorization.authorization_id},
        )
        self.store.clear_lock("mainnet_authorization")
        return authorization.model_dump(mode="json") | {
            "capability": self.capabilities.status(authorization.authorization_id),
            "wallet": wallet_status,
        }

    def mainnet_authorization_status(self) -> dict[str, Any]:
        pointer = self.store.get("pending_mainnet_authorization")
        if not isinstance(pointer, dict) or not pointer.get("authorization_id"):
            return {"available": False, "locked": True}
        authorization_id = str(pointer["authorization_id"])
        row = self.store.authorization(authorization_id)
        try:
            capability = self.capabilities.status(authorization_id)
        except SecretStoreError as exc:
            capability = {
                "authorization_id": authorization_id,
                "available": False,
                "error": str(exc),
            }
        expired = True
        if row and row.get("expires_at"):
            try:
                expired = datetime.fromisoformat(str(row["expires_at"])) <= datetime.now(UTC)
            except ValueError:
                expired = True
        available = bool(
            row
            and not row.get("consumed_at")
            and not row.get("revoked_at")
            and not expired
            and capability.get("available")
        )
        return {
            "available": available,
            "locked": not available,
            "expired": expired,
            "authorization": row,
            "capability": capability,
        }

    def revoke_mainnet_authorization(self, authorization_id: str) -> dict[str, Any]:
        revoked = self.store.revoke_mainnet_authorization(authorization_id)
        self.capabilities.delete(authorization_id)
        pointer = self.store.get("pending_mainnet_authorization")
        if isinstance(pointer, dict) and pointer.get("authorization_id") == authorization_id:
            self.store.set("pending_mainnet_authorization", None)
        self.store.latch_lock(
            "mainnet_authorization",
            reason="Capability mainnet ausente ou revogada",
            payload={"authorization_id": authorization_id},
        )
        return {"authorization_id": authorization_id, "revoked": revoked}

    def testnet_certificate_status(self) -> dict[str, Any]:
        certificate = self.store.latest_testnet_certificate()
        if certificate is None:
            return {"available": False}
        return certificate.model_dump(mode="json") | {
            "available": True,
            "valid": certificate.verify_content_address(),
        }

    def _consume_mainnet_capability(
        self,
        *,
        release: ReleaseManifest,
        budget_usdc: float,
        account_address: str,
        wallet_address: str,
        session_id: str,
    ) -> MainnetAuthorization:
        pointer = self.store.get("pending_mainnet_authorization")
        if not isinstance(pointer, dict) or not pointer.get("authorization_id"):
            self.store.latch_lock(
                "mainnet_authorization",
                reason="Capability mainnet de uso único ausente",
            )
            raise ControlError("Mainnet bloqueada: capability DPAPI ausente")
        authorization_id = str(pointer["authorization_id"])
        try:
            capability = self.capabilities.load(authorization_id)
            token_hash = sha256(str(capability["token"]).encode()).hexdigest()
            certificate = self.store.testnet_certificate(str(capability["certificate_id"]))
            if certificate is None or not certificate.verify_content_address():
                raise ValueError("Certificado testnet inválido")
            bindings = _mainnet_release_bindings(release)
            expected_capability: dict[str, object] = {
                "release_id": release.release_id,
                "certificate_id": certificate.certificate_id,
                "model_hash": bindings["model_hash"],
                "gates_hash": bindings["gates_hash"],
                "git_commit": bindings["git_commit"],
                "config_sha256": bindings["config_sha256"],
                "config_fingerprint": bindings["config_fingerprint"],
                "code_hash": bindings["code_hash"],
                "account_address": account_address.lower(),
                "wallet_address": wallet_address.lower(),
            }
            for key, expected in expected_capability.items():
                actual = capability.get(key)
                if "address" in key:
                    actual = str(actual).lower()
                if actual != expected:
                    raise ValueError(f"Binding DPAPI divergente: {key}")
            if abs(float(capability.get("budget_usdc", 0)) - budget_usdc) > 1e-9:
                raise ValueError("Binding DPAPI divergente: budget_usdc")
            authorization = self.store.consume_mainnet_authorization(
                authorization_id,
                token_hash=token_hash,
                session_id=session_id,
                release_id=release.release_id,
                certificate_id=certificate.certificate_id,
                model_hash=bindings["model_hash"],
                gates_hash=bindings["gates_hash"],
                git_commit=bindings["git_commit"],
                config_sha256=bindings["config_sha256"],
                config_fingerprint=bindings["config_fingerprint"],
                code_hash=bindings["code_hash"],
                account_address=account_address,
                wallet_address=wallet_address,
                budget_usdc=budget_usdc,
            )
        except Exception as exc:
            self.store.latch_lock(
                "mainnet_authorization",
                reason="Capability mainnet inválida, expirada ou divergente",
                payload={"authorization_id": authorization_id, "error": str(exc)},
            )
            raise ControlError(f"Mainnet bloqueada: {exc}") from exc
        self.capabilities.delete(authorization_id)
        self.store.set("pending_mainnet_authorization", None)
        self.store.clear_lock("mainnet_authorization")
        return authorization

    def _assert_secret_mutation_allowed(self, environment: Environment) -> None:
        if environment is Environment.PAPER:
            raise ControlError("Paper não possui cofre de API wallet")
        state = self.store.agent_state()
        if state.status is not AgentStatus.DISARMED:
            raise ControlError("Geração/rotação de segredo exige agente desarmado")

    def _assert_testnet_operation_allowed(self) -> None:
        state = self.store.agent_state()
        if state.environment is Environment.MAINNET or (
            state.environment is not Environment.TESTNET
            and state.status is not AgentStatus.DISARMED
        ):
            raise ControlError(
                "Operação testnet recusada enquanto outra sessão/ambiente está ativo"
            )


def gate_paths_from_reports(paths: AppPaths) -> list[Path]:
    return sorted(
        paths.reports.glob("*/gate.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings_for_key(value: object, key: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item_key, item in value.items():
            if item_key == key:
                if isinstance(item, str):
                    found.add(item)
                elif isinstance(item, list):
                    found.update(str(entry) for entry in item if isinstance(entry, str))
            found.update(_strings_for_key(item, key))
    elif isinstance(value, list):
        for item in value:
            found.update(_strings_for_key(item, key))
    return found


def _snapshot_counts(snapshot: AccountSnapshot) -> dict[str, object]:
    return {
        "equity": snapshot.equity,
        "positions": len(snapshot.positions),
        "orders": len(snapshot.open_orders),
    }


def _release_bindings(release: ReleaseManifest) -> tuple[str, str]:
    try:
        return (
            str(release.artifacts["model"]["sha256"]),
            str(release.artifacts["research_approval"]["sha256"]),
        )
    except KeyError as exc:
        raise ControlError("Release não possui bindings de modelo/gates") from exc


def _mainnet_release_bindings(release: ReleaseManifest) -> dict[str, str]:
    model_hash, gates_hash = _release_bindings(release)
    try:
        config_sha256 = str(release.artifacts["config"]["sha256"])
        gate_path = Path(release.artifacts["research_approval"]["path"])
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        binding = gate["release_binding"]
        git_commit = str(release.git_commit)
        config_fingerprint = str(binding["config_fingerprint"])
        code_hash = str(binding["code_hash"])
        gate_commit = str(binding["commit_hash"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ControlError("Release lacks mainnet commit/configuration/code bindings") from exc
    values = {
        "model_hash": model_hash,
        "gates_hash": gates_hash,
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "config_fingerprint": config_fingerprint,
        "code_hash": code_hash,
    }
    if gate_commit != git_commit or any(not value for value in values.values()):
        raise ControlError("Release/gate mainnet binding is empty or divergent")
    return values


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return sha256(encoded).hexdigest()
