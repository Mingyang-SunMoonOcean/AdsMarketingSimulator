from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .market_physics import MarketPhysics
from .state_manager import StateManager
from .volatility_scheduler import VolatilityScheduler
from .world_clock import WorldClock


class SandboxEnv:
    """
    Wrapper / interaction surface for agents and APIs.
    This is the single entry point for observing state and taking actions.

    Owns the shared ``WorldClock`` that every sub-system reads from.

    Pass ``seed`` (int or None) to fully reproducible stochastic market draws.
    The seed is forwarded to ``MarketPhysics``; if an explicit ``physics``
    instance is already provided, ``seed`` is ignored.
    """

    def __init__(
        self,
        physics: MarketPhysics | None = None,
        state: StateManager | None = None,
        scheduler: VolatilityScheduler | None = None,
        clock: WorldClock | None = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.clock = clock or WorldClock()
        self.state = state or StateManager(clock=self.clock)
        self.physics = physics or MarketPhysics(seed=seed)
        self.scheduler = scheduler or VolatilityScheduler()

    def observe(self) -> list:
        return self.state.observe()

    # ------------------------------------------------------------------
    # Agent-facing API surface
    # ------------------------------------------------------------------
    def configure(self, daily_budget: float | None = None, max_bid: float | None = None) -> None:
        """
        Operating agents use this to **write** configuration to the StateManager.
        """
        if daily_budget is not None:
            self.state.set_daily_budget(daily_budget)
        if max_bid is not None:
            self.state.set_max_bid(max_bid)

    def act(self) -> dict:
        """
        Advance the simulation by one step.

        - VolatilityScheduler **writes** ν into the StateManager.
        - MarketPhysics **reads** budget / max_bid / ν from the StateManager
          and **writes** outcomes back (with timestamps every 15 minutes).
        - The WorldClock is advanced inside ``StateManager.record_outcome()``.
        - The latest outcome is returned to the caller.
        """
        # 1) VolatilityScheduler updates ν in the state (reads time from clock).
        current_hour = self.clock.current_hour
        sched_result = self.scheduler.get_v_multiplier(current_hour)

        # Support both legacy float return values and the new dict format
        # {"v": float, "event": str}.
        v = float(sched_result["v"])

        self.state.set_volatility(v)

        # 2) MarketPhysics consumes state and records outcomes via StateManager.
        outcome = self.physics.run_step(self.state)
        return outcome
