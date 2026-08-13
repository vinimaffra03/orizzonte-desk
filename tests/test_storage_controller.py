from __future__ import annotations

import pytest

from orizzonte_desk.controller import AgentController, ControlError
from orizzonte_desk.models import AgentStatus, Environment
from orizzonte_desk.storage import StateStore


def test_store_uses_wal_and_roundtrips_state(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    state = store.agent_state()
    assert state.status is AgentStatus.DISARMED
    store.event("test", "evento")
    assert store.recent_events(1)[0]["message"] == "evento"


def test_paper_arm_requires_exact_confirmation(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)
    expected = controller.expected_confirmation(Environment.PAPER, 1000)
    armed = controller.arm(
        environment=Environment.PAPER,
        budget_usdc=1000,
        confirmation=expected,
    )
    assert armed.status is AgentStatus.ARMED
    running = controller.start()
    assert running.status is AgentStatus.RUNNING
    paused = controller.pause()
    assert paused.status is AgentStatus.PAUSED
    assert controller.start().status is AgentStatus.RUNNING
    assert controller.flatten().status is AgentStatus.LOCKED
    assert controller.disarm().status is AgentStatus.DISARMED


def test_control_rejects_invalid_transitions(app_paths, settings) -> None:
    store = StateStore(app_paths.database)
    store.initialize()
    controller = AgentController(app_paths, settings, store)

    with pytest.raises(ControlError, match="iniciar"):
        controller.start()
    with pytest.raises(ControlError, match="positivo"):
        controller.arm(
            environment=Environment.PAPER,
            budget_usdc=0,
            confirmation=controller.expected_confirmation(Environment.PAPER, 0),
        )
    with pytest.raises(ControlError, match="Confirma"):
        controller.arm(
            environment=Environment.PAPER,
            budget_usdc=1000,
            confirmation="NUNCA OPERAR",
        )
