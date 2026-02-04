from __future__ import annotations

from market_physics import MarketPhysics
from state_manager import StateManager
from volatility_scheduler import VolatilityScheduler


class SandboxEnv:
    """
    Wrapper / interaction surface for agents and APIs.
    This is the single entry point for observing state and taking actions.
    """

    def __init__(
        self,
        physics: MarketPhysics | None = None,
        state: StateManager | None = None,
        scheduler: VolatilityScheduler | None = None,
    ):
        self.state = state or StateManager()
        self.physics = physics or MarketPhysics()
        self.scheduler = scheduler or VolatilityScheduler()

    def observe(self) -> dict:
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
        - The latest outcome is returned to the caller.
        """
        # 1) VolatilityScheduler updates ν in the state.
        current_hour = self.state.current_hour
        sched_result = self.scheduler.get_v_multiplier(current_hour)

        # Support both legacy float return values and the new dict format
        # {"v": float, "event": str}.
        v = float(sched_result["v"])

        self.state.set_volatility(v)

        # 2) MarketPhysics consumes state and records outcomes via StateManager.
        outcome = self.physics.run_step(self.state)
        return outcome

