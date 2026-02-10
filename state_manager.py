from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

from world_clock import WorldClock

# Budget pacing constants
STEPS_PER_HOUR = 4          # 60 min / 15 min per step
ROLLING_24H_STEPS = 96      # 24 hours * 4 steps/hour

@dataclass
class BizInputs:
    daily_budget: float = 0.0
    max_bid: float = 0.0

@dataclass
class ExternalEventsInputs:
    volatility: float = 1.0

@dataclass
class MarketOutcome:
    #timestamp
    current_minute: int = 0
    current_hour: int = 0
    current_day: int = 0
    clicks: int = 0
    leads: int = 0
    spend: float = 0.0
    cpa: float = 0.0
    realized_cpc: float = 0.0

@dataclass
class DerivedVariables:
    budget_status: str = "normal"
    current_day_spend: float = 0.0

@dataclass
class SimulationState:
    """
    Core simulation state, managed centrally by StateManager.

    Time is **not** stored here -- it lives in the shared ``WorldClock``.
    """
    biz_inputs: BizInputs = field(default_factory=BizInputs)
    external_events_inputs: ExternalEventsInputs = field(default_factory=ExternalEventsInputs)
    market_outcome: MarketOutcome = field(default_factory=MarketOutcome)
    derived_variables: DerivedVariables = field(default_factory=DerivedVariables)

class StateManager:
    """
    Single source of truth for all simulation state.

    - Agents **write** configuration (budget / max bid) here.
    - VolatilityScheduler **writes** the current volatility multiplier here.
    - MarketPhysics **reads** inputs from here and **writes** outcomes back.
    - Outcomes are timestamped and appended to an in-memory state history.
    - When an agent "observes" the environment, the full state history is returned.
    """

    def __init__(
        self,
        clock: WorldClock,
        daily_budget: float = 1000.0,
    ) -> None:
        self.clock = clock

        self.state = SimulationState(
            biz_inputs=BizInputs(daily_budget=float(daily_budget)),
        )
        self.state_history: list[SimulationState] = []

    # ------------------------------------------------------------------
    # Configuration writes (from operating agent via API)
    # ------------------------------------------------------------------
    def set_daily_budget(self, daily_budget: float) -> None:
        self.state.biz_inputs.daily_budget = float(daily_budget)

    def set_max_bid(self, max_bid: float) -> None:
        self.state.biz_inputs.max_bid = float(max_bid)

    # ------------------------------------------------------------------
    # Volatility writes (from VolatilityScheduler via SandboxEnv)
    # ------------------------------------------------------------------
    def set_volatility(self, v_multiplier: float) -> None:
        self.state.external_events_inputs.volatility = float(v_multiplier)

    # ------------------------------------------------------------------
    # Reads used by MarketPhysics / VolatilityScheduler
    # ------------------------------------------------------------------
    def get_inputs(self) -> Tuple[BizInputs, ExternalEventsInputs]:
        """
        Returns the current biz inputs and external event inputs
        used by the physics engine.
        """
        return self.state.biz_inputs, self.state.external_events_inputs

    @property
    def current_hour(self) -> int:
        """Delegate to the shared WorldClock."""
        return self.clock.current_hour

    @property
    def current_day(self) -> int:
        """Delegate to the shared WorldClock."""
        return self.clock.current_day

    # ------------------------------------------------------------------
    # Outcomes written by MarketPhysics
    # ------------------------------------------------------------------
    def record_outcome(self, outcome: dict[str, Any]) -> None:
        """
        Record a single step outcome.

        - Stamps it with time (minute/hour/day) from the shared WorldClock.
        - Wraps raw outcome values in a ``MarketOutcome`` dataclass.
        - Updates ``DerivedVariables`` (budget_status, current_day_spend).
        - Creates a **snapshot** of the full ``SimulationState`` and appends
          it to ``state_history``.
        - Advances the WorldClock by one tick.

        Budget model:
        - Budget is checked per calendar day against ``daily_budget``.
        """
        minute = self.clock.current_minute
        hour = self.clock.current_hour
        day = self.clock.current_day
        daily_budget = self.state.biz_inputs.daily_budget

        # Current calendar-day spend (including this step)
        current_day_spend = sum(
            s.market_outcome.spend
            for s in self.state_history
            if s.market_outcome.current_day == day
        )
        step_spend = float(outcome.get("spend", 0.0))
        current_day_spend += step_spend

        # Determine budget status for the current day
        if current_day_spend >= daily_budget:
            budget_status = "budget_depleted"
        elif current_day_spend >= 0.9 * daily_budget:
            budget_status = "budget_constrained"
        else:
            budget_status = "normal"

        # Build typed MarketOutcome
        market_outcome = MarketOutcome(
            current_minute=minute,
            current_hour=hour,
            current_day=day,
            clicks=int(outcome.get("clicks", 0)),
            leads=int(outcome.get("leads", 0)),
            spend=round(step_spend, 4),
            cpa=round(float(outcome.get("cpa", 0.0)), 4),
            realized_cpc=round(float(outcome.get("realized_cpc", 0.0)), 4),
        )

        # Update live state
        self.state.market_outcome = market_outcome
        self.state.derived_variables.budget_status = budget_status
        self.state.derived_variables.current_day_spend = round(current_day_spend, 4)

        # Append an immutable *snapshot* so history entries don't alias
        snapshot = SimulationState(
            biz_inputs=BizInputs(
                daily_budget=self.state.biz_inputs.daily_budget,
                max_bid=self.state.biz_inputs.max_bid,
            ),
            external_events_inputs=ExternalEventsInputs(
                volatility=self.state.external_events_inputs.volatility,
            ),
            market_outcome=market_outcome,
            derived_variables=DerivedVariables(
                budget_status=budget_status,
                current_day_spend=round(current_day_spend, 4),
            ),
        )
        self.state_history.append(snapshot)

        # Advance the shared world clock
        self.clock.tick()

    # ------------------------------------------------------------------
    # Reads used by operating agent (API)
    # ------------------------------------------------------------------
    def observe(self) -> list[SimulationState]:
        """
        Return the full state history for agents.

        Each entry is a ``SimulationState`` snapshot taken at the end of
        that step.  The last element is the most recent.
        """
        return self.state_history

