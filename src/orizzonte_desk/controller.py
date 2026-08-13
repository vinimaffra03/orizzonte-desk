from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from orizzonte_desk.config import Settings
from orizzonte_desk.constants import LIVE_CONFIRMATION_PREFIX
from orizzonte_desk.exchange import HyperliquidGateway, PaperGateway, TradingGateway
from orizzonte_desk.models import AgentState, AgentStatus, Environment
from orizzonte_desk.paths import AppPaths
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
        self._paper_gateway = PaperGateway(settings.backtest.initial_capital)

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
        expected = self.expected_confirmation(environment, budget_usdc)
        if confirmation != expected:
            raise ControlError(f"Confirmação inválida. Digite exatamente: {expected}")
        self.paths.assert_free_space(self.settings.app.minimum_free_gb)
        gateway = self.gateway(environment)
        if environment is not Environment.PAPER:
            self._validate_research_approval()
            self._validate_promoted_model()
        snapshot = gateway.snapshot()
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
            metadata={"preflight_equity": snapshot.equity, "isolated_leverage": 10},
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
        running = state.model_copy(
            update={"status": AgentStatus.RUNNING, "last_heartbeat": datetime.now(UTC)}
        )
        self.store.save_agent_state(running)
        self.store.event("control", "Agente iniciado")
        return running

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
        )

    def _validate_research_approval(self) -> None:
        approval = self.paths.reports / "live-approval.json"
        if not approval.exists():
            raise ControlError("Gate combinado ausente: execute `orizzonte backtest compare`")
        payload = json.loads(approval.read_text(encoding="utf-8"))
        if not payload.get("passed", False):
            raise ControlError("Gate combinado de pesquisa está reprovado")

    def _validate_promoted_model(self) -> None:
        promoted = self.paths.models / "promoted.json"
        model = self.paths.models / "promoted.joblib"
        if not promoted.exists() or not model.exists():
            raise ControlError("Nenhum modelo promovido está disponível")


def gate_paths_from_reports(paths: AppPaths) -> list[Path]:
    return sorted(
        paths.reports.glob("*/gate.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )
