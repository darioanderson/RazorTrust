from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .domain import HoldState
from .repository import HoldRepository, InvalidTransition
from .workflow import HoldService


@dataclass(frozen=True, slots=True)
class AutomationCycleResult:
    scanned: int = 0
    evaluated: int = 0
    skipped_missing_input: int = 0
    skipped_non_open: int = 0
    failed: int = 0


class HoldAutomationOrchestrator:
    """Safely auto-evaluate already-created OPEN holds with persisted evaluation input.

    This orchestrator never moves money. In HUMAN_ONLY mode the HoldService can only route
    cases into the human-gated path. Missing feature/evaluation input is skipped rather than
    fabricated.
    """

    def __init__(self, repository: HoldRepository, service: HoldService) -> None:
        self.repository = repository
        self.service = service

    async def run_once(self, *, limit: int = 200) -> AutomationCycleResult:
        holds = await self.repository.list_holds(merchant_id=None, limit=limit)
        scanned = evaluated = skipped_missing_input = skipped_non_open = failed = 0
        for hold in holds:
            scanned += 1
            if hold.state != HoldState.OPEN:
                skipped_non_open += 1
                continue
            try:
                evaluation_input = await self.repository.get_evaluation_input(hold.hold_id)
            except (InvalidTransition, KeyError):
                skipped_missing_input += 1
                continue
            try:
                await self.service.evaluate(hold.hold_id, evaluation_input)
                evaluated += 1
            except Exception:
                # The decision service itself fails closed; the orchestrator keeps the loop alive.
                failed += 1
        return AutomationCycleResult(
            scanned=scanned,
            evaluated=evaluated,
            skipped_missing_input=skipped_missing_input,
            skipped_non_open=skipped_non_open,
            failed=failed,
        )

    async def run_forever(self, *, interval_seconds: int, stop_event: asyncio.Event) -> None:
        if interval_seconds < 5:
            raise ValueError("automation interval must be at least 5 seconds")
        while not stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except TimeoutError:
                continue
