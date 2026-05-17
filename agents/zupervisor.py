"""Human-analogue supervision logic for OODA MAS."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.state_manager import SimulationState
from core.volatility_scheduler import WEBSITE_CRASH_HOURS, HOLIDAY_HOURS

# Mirrors legacy_human constants — Zupervisor uses the same economic targets.
TARGET_CPA:           float = 80.0
CPA_HIGH_THRESHOLD:   float = TARGET_CPA * 1.2   # 96.0 — cut budget if exceeded
CPA_LOW_THRESHOLD:    float = TARGET_CPA * 0.8   # 64.0 — scale budget if below
BUDGET_CUT_FACTOR:    float = 0.70               # −30 % (legacy_human: −30 %)
BUDGET_SCALE_FACTOR:  float = 1.20               # +20 % (legacy_human: +20 %)

# Baseline-aligned budget trigger constants (legacy_human parity).
CRASH_END_HOUR = WEBSITE_CRASH_HOURS[1]
HOLIDAY_START_HOUR = HOLIDAY_HOURS[0]
HOLIDAY_END_HOUR = HOLIDAY_HOURS[1]
INITIAL_DAILY_BUDGET = 1000.0
# Higher than IB's nominal CHF 1000 holiday cap so MAS can sustain competitive bids
# through full surge days without repeated budget-depletion throttles.
HOLIDAY_DAILY_BUDGET = 1650.0


class Zupervisor:
    """
    Zupervisor is the MAS equivalent of the human intervener.

    Budget triggers mirror baseline/legacy_human.py exactly. MAS may call this
    checker frequently, so trigger evaluation is de-duplicated to once per hour.
    """

    def __init__(
        self,
        policy_path: str = "agents/knowledge/policy_db.json",
        shared_kb_path: str = "agents/knowledge/shared_knowledge.json",
        log_path: str = "agents/logs/zupervisor_log.jsonl"
    ):
        self.policy_path   = policy_path
        self.shared_kb_path = shared_kb_path
        self.log_path       = log_path
        self._last_weekly_review_hour: Optional[int] = None
        self._last_holiday_prep_hour: Optional[int] = None
        self._last_holiday_reset_hour: Optional[int] = None
        self._last_recovery_safety_hour: Optional[int] = None
        self._last_evaluated_hour: Optional[int] = None
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _weekly_cpa(state_history: List[SimulationState]) -> float:
        """
        Weekly CPA over the latest 7-day window, mirroring legacy_human section B.
        """
        weekly_history = state_history[-672:]
        total_spend = sum(s.market_outcome.spend for s in weekly_history)
        total_leads = sum(s.market_outcome.leads for s in weekly_history)
        return total_spend / total_leads if total_leads > 0 else float("inf")

    @staticmethod
    def _window_overlaps_range(
        window_start_hour: int,
        window_end_hour: int,
        range_start_hour: int,
        range_end_hour: int,
    ) -> bool:
        """True when [window_start_hour, window_end_hour] intersects [range_start_hour, range_end_hour]."""
        return not (window_end_hour < range_start_hour or window_start_hour > range_end_hour)

    def evaluate_budget_triggers(
        self,
        current_hour: int,
        scheduler_event: str,
        state_history: List[SimulationState],
    ) -> Optional[Dict[str, Any]]:
        """
        Baseline-equivalent budget trigger evaluator.
        """
        if not state_history:
            return None
        # MAS may invoke this each 15-min step; baseline semantics are hourly.
        if current_hour == self._last_evaluated_hour:
            return None
        self._last_evaluated_hour = current_hour

        latest = state_history[-1]
        current_budget = float(latest.biz_inputs.daily_budget)
        suggested_budget = current_budget
        reasons: List[str] = []

        # --- B. Weekly Efficiency Review (every 168h / 7 days) ---
        if (
            current_hour > 0
            and current_hour % 168 == 0
            and current_hour != self._last_weekly_review_hour
        ):
            weekly_start = current_hour - 167
            weekly_end = current_hour
            overlaps_crash = self._window_overlaps_range(
                weekly_start, weekly_end, WEBSITE_CRASH_HOURS[0], WEBSITE_CRASH_HOURS[1]
            )
            # Avoid suppressing holiday capture with weekly cuts when crash contamination
            # or in-holiday windows distort the weekly CPA signal.
            overlaps_holiday = self._window_overlaps_range(
                weekly_start, weekly_end, HOLIDAY_HOURS[0], HOLIDAY_HOURS[1]
            )
            weekly_cpa = self._weekly_cpa(state_history)
            if weekly_cpa > TARGET_CPA * 1.2 and not (overlaps_crash or overlaps_holiday):
                suggested_budget = current_budget * BUDGET_CUT_FACTOR
                reasons.append(
                    f"weekly_efficiency_cut: weekly CPA {weekly_cpa:.2f} > {TARGET_CPA * 1.2:.2f}"
                )
            elif weekly_cpa < TARGET_CPA * 0.8:
                suggested_budget = current_budget * BUDGET_SCALE_FACTOR
                reasons.append(
                    f"weekly_efficiency_scale: weekly CPA {weekly_cpa:.2f} < {TARGET_CPA * 0.8:.2f}"
                )
            self._last_weekly_review_hour = current_hour

        # --- C. Event-based recovery budget reset (2h after crash end) ---
        # One-shot trigger: avoid repeatedly forcing 50% budget while bids recover.
        if (
            scheduler_event == "NORMAL"
            and current_hour == (CRASH_END_HOUR + 2)
            and current_hour != self._last_recovery_safety_hour
            and latest.biz_inputs.max_bid < 0.50
        ):
            suggested_budget = INITIAL_DAILY_BUDGET * 0.5
            reasons.append(
                "recovery_safety_budget_once: one-shot crash recovery safeguard to 50% budget"
            )
            self._last_recovery_safety_hour = current_hour

        # --- D1. Holiday prep (24h before holiday start) ---
        prep_hour = HOLIDAY_START_HOUR - 24
        if current_hour == prep_hour and current_hour != self._last_holiday_prep_hour:
            suggested_budget = HOLIDAY_DAILY_BUDGET
            reasons.append("holiday_prep: set holiday budget 24h before start")
            self._last_holiday_prep_hour = current_hour

        # --- D2. Holiday cooldown reset (12h after holiday end) ---
        reset_hour = HOLIDAY_END_HOUR + 12
        if current_hour == reset_hour and current_hour != self._last_holiday_reset_hour:
            suggested_budget = INITIAL_DAILY_BUDGET
            reasons.append("holiday_reset: reset baseline budget after holiday cooldown")
            self._last_holiday_reset_hour = current_hour

        suggested_budget = round(suggested_budget, 2)
        if abs(suggested_budget - current_budget) <= 0.01:
            return None

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "budget_trigger",
            "virtual_hour": current_hour,
            "virtual_day": latest.market_outcome.current_day,
            "scheduler_event": scheduler_event,
            "current_budget": round(current_budget, 2),
            "suggested_daily_budget": suggested_budget,
            # Pair the one-shot recovery safety budget with a bid restoration
            # so the campaign exits sub-competitive silence immediately.
            "suggested_max_bid": (
                5.0
                if (
                    current_hour == (CRASH_END_HOUR + 2)
                    and "recovery_safety_budget_once: one-shot crash recovery safeguard to 50% budget" in reasons
                )
                else None
            ),
            "reasons": reasons,
        }
        self._log_guidance(report)
        return report

    # --------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------

    def provide_guidance(
        self,
        current_day: int,
        state_history: List[SimulationState],
    ) -> Dict[str, Any]:
        """
        Strategic heartbeat only.

        Human baseline does not tune alpha/beta from this path, so no policy
        mutations are performed here.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "day": current_day,
            "action": "MONITOR",
            "reasoning": "Strategic heartbeat. Budget updates are handled by evaluate_budget_triggers().",
        }

        self._log_guidance(report)
        return report

    def _log_guidance(self, report: Dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(report) + "\n")
