from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import LIVE_CONFIRMATION_PREFIX
from orizzonte_desk.exchange import (
    AccountSnapshot,
    HyperliquidGateway,
    PaperGateway,
    TradingGateway,
)
from orizzonte_desk.models import AgentState, AgentStatus, Environment, Side, Signal
from orizzonte_desk.paths import AppPaths
from orizzonte_desk.release import ReleaseError, ReleaseManager, ReleaseManifest
from orizzonte_desk.secrets import DPAPISecretStore
from orizzonte_desk.storage import StateStore


class ControlError(RuntimeError):
    pass


class AgentController:
    def __init__(self, paths: AppPaths, settings: Settings, store: StateStore) -> None:
        self.paths = paths
        self.settings = settings
        self.store = store
        self.secrets = DPAPISecretStore(paths.secret_file)
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
        if environment is Environment.MAINNET:
            raise ControlError(
                "Mainnet está bloqueada nesta versão; somente paper e testnet são autorizados"
            )
        expected = self.expected_confirmation(environment, budget_usdc)
        if confirmation != expected:
            raise ControlError(f"Confirmação inválida. Digite exatamente: {expected}")
        self.paths.assert_free_space(self.settings.app.minimum_free_gb)
        gateway = self.gateway(environment)
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
        if environment is not Environment.PAPER:
            account = str(self.secrets.load()["account_address"]).lower()
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
                "release_id": (
                    release.release_id if environment is not Environment.PAPER else None
                ),
                "reconciled_at": datetime.now(UTC).isoformat(),
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
        if state.environment is Environment.MAINNET:
            raise ControlError("Mainnet está permanentemente bloqueada nesta versão")
        self._validate_research_approval()
        self._validate_promoted_model()
        release = self._validate_approved_release()
        if state.metadata.get("release_id") != release.release_id:
            raise ControlError("Release da sessão mudou; desarme e arme novamente")
        self._assert_no_persistent_lock()
        known_symbols = {item["symbol"] for item in self.store.positions()}
        known_cloids = {item["client_order_id"] for item in self.store.orders()}
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
        state.metadata["preflight"] = preflight
        state.metadata["reconciled_at"] = datetime.now(UTC).isoformat()
        state.metadata.pop("restart_reconcile_required", None)
        self.store.save_agent_state(state)

    def pause(self) -> AgentState:
        state = self.store.agent_state()
        if state.status is not AgentStatus.RUNNING:
            raise ControlError("Somente um agente em execução pode ser pausado")
        paused = state.model_copy(update={"status": AgentStatus.PAUSED})
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
        payload = self.secrets.load()
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
        return {
            "environment": Environment.TESTNET.value,
            "equity": snapshot.equity,
            "positions": len(snapshot.positions),
            "orders": len(snapshot.open_orders),
            "fills_persisted": len(self.store.fills()),
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
            gateway.schedule_dead_man(30)
            lifecycle["entry"] = gateway.place_entry_with_protection(
                signal,
                size=size,
                stop_price=price * 0.99,
                take_profit_price=price * 1.01,
                slippage=0.01,
            )
            lifecycle["after_entry"] = _snapshot_counts(gateway.reconcile())
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
        self.store.event("testnet", "Smoke testnet concluído", payload=lifecycle)
        return lifecycle

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
