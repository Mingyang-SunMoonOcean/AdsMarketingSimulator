from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any


# Budget pacing constants
STEPS_PER_HOUR = 4          # 60 min / 15 min per step
ROLLING_24H_STEPS = 96      # 24 hours * 4 steps/hour

@dataclass
class SimulationState:
    """
    Core simulation state, managed centrally by StateManager.
    Time is tracked in minutes, with each step advancing by a fixed interval.
    """

    current_minute: int = 0
    daily_budget: float = 0.0
    max_bid: float = 0.0
    volatility: float = 1.0
    step_index: int = 0
    latest_outcome: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    budget_status: str = "normal"

class StateManager:
    """
    Single source of truth for all simulation state.

    - Agents **write** configuration (budget / max bid) here.
    - VolatilityScheduler **writes** the current volatility multiplier here.
    - MarketPhysics **reads** inputs from here and **writes** outcomes back.
    - Outcomes are timestamped in minutes (every `step_minutes`, starting from 0)
      and appended to an in-memory history and a local CSV "database".
    - When an agent "observes" the environment, the latest outcome is returned.
    """

    def __init__(
        self,
        daily_budget: float = 1000.0,
        step_minutes: int = 15,
        minutes_per_day: int = 60 * 24,
        results_csv_path: str = "simulation_results.csv",
    ) -> None:
        self.step_minutes = int(step_minutes)
        self.minutes_per_day = int(minutes_per_day)
        self.results_csv_path = results_csv_path

        # Optionally reset the results CSV at the start of a new run.
        # This ensures that each simulation instance writes a fresh timeline.
        if os.path.exists(self.results_csv_path):
            os.remove(self.results_csv_path)

        self.state = SimulationState(daily_budget=float(daily_budget))

    # ------------------------------------------------------------------
    # Configuration writes (from operating agent via API)
    # ------------------------------------------------------------------
    def set_daily_budget(self, daily_budget: float) -> None:
        self.state.daily_budget = float(daily_budget)

    def set_max_bid(self, max_bid: float) -> None:
        self.state.max_bid = float(max_bid)

    # ------------------------------------------------------------------
    # Volatility writes (from VolatilityScheduler via SandboxEnv)
    # ------------------------------------------------------------------
    def set_volatility(self, v_multiplier: float) -> None:
        self.state.volatility = float(v_multiplier)

    # ------------------------------------------------------------------
    # Reads used by MarketPhysics / VolatilityScheduler
    # ------------------------------------------------------------------
    def get_inputs(self) -> dict[str, Any]:
        """
        Returns the current configuration and time snapshot used by the physics.
        """
        day = self.current_day
        cumulative_budget = self.state.daily_budget * day
        total_spend = sum(float(r.get("spend", 0.0)) for r in self.state.history)

        return {
            "minute": self.state.current_minute,
            "hour": self.current_hour,
            "day": day,
            "daily_budget": self.state.daily_budget,
            "cumulative_budget": cumulative_budget,
            "total_spend": total_spend,
            "max_bid": self.state.max_bid,
            "volatility": self.state.volatility,
        }

    @property
    def current_hour(self) -> int:
        return self.state.current_minute // 60

    @property
    def current_day(self) -> int:
        # Day index starting from 0; expose as 1-based to callers.
        return (self.state.current_minute // self.minutes_per_day) + 1

    # ------------------------------------------------------------------
    # Outcomes written by MarketPhysics
    # ------------------------------------------------------------------
    def record_outcome(self, outcome: dict[str, Any]) -> None:
        """
        Record a single step outcome.

        - Stamps it with time (minute/hour/day) and a monotonically increasing step index.
        - Stores in memory.
        - Appends to a local CSV file.
        - Advances simulation time by `step_minutes`.

        Budget model:
        - Cumulative budget grows by `daily_budget` every 24 hours.
        - Pacing is checked against a rolling 24-hour spend window.
        """
        minute = self.state.current_minute
        hour = self.current_hour
        day = self.current_day
        daily_budget = self.state.daily_budget

        # Cumulative budget: grows by daily_budget each day
        cumulative_budget = daily_budget * day

        # Total spend across the entire simulation
        total_spend = sum(float(r.get("spend", 0.0)) for r in self.state.history)
        total_spend += float(outcome.get("spend", 0.0))

        # Rolling 24-hour spend (for pacing visibility)
        rolling_24h_spend = sum(
            float(r.get("spend", 0.0))
            for r in self.state.history[-ROLLING_24H_STEPS:]
        )
        rolling_24h_spend += float(outcome.get("spend", 0.0))

        # Determine budget status based on cumulative budget
        if total_spend >= cumulative_budget:
            budget_status = "budget_depleted"
        elif total_spend >= 0.9 * cumulative_budget:
            budget_status = "budget_constrained"
        else:
            budget_status = "normal"

        record: dict[str, Any] = {
            "step_index": self.state.step_index,
            "minute": minute,
            "hour": hour,
            "day": day,
            "daily_budget": self.state.daily_budget,
            "cumulative_budget": round(cumulative_budget, 4),
            "total_spend": round(total_spend, 4),
            "rolling_24h_spend": round(rolling_24h_spend, 4),
            "max_bid": self.state.max_bid,
            "volatility": self.state.volatility,
            "budget_status": budget_status,
        }
        record.update(outcome)

        # Update in-memory state
        self.state.latest_outcome = record
        self.state.history.append(record)
        self.state.step_index += 1
        self.state.current_minute += self.step_minutes
        self.state.budget_status = budget_status


        # Persist to CSV "database"
        self._append_to_csv(record)

    def _append_to_csv(self, record: dict[str, Any]) -> None:
        """
        Append a single record to the CSV file, creating it with a header
        if it does not yet exist.
        """
        file_exists = os.path.exists(self.results_csv_path)

        # Ensure deterministic column order
        fieldnames = list(record.keys())

        with open(self.results_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    # ------------------------------------------------------------------
    # Reads used by operating agent (API)
    # ------------------------------------------------------------------
    def observe(self) -> dict[str, Any]:
        """
        Return the latest observable state for agents.

        - Always includes current time and config fields.
        - If no outcome has been produced yet, `latest_outcome` will be None.
        """
        base = self.get_inputs()
        base["step_index"] = self.state.step_index
        base["latest_outcome"] = self.state.latest_outcome
        base["history"] = self.state.history
        return base

