"""Deterministic execution / guardrail layer (ACT phase) for OODA MAS."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from core.state_manager import SimulationState
from core.volatility_scheduler import HOLIDAY_HOURS

class Taskmaster:
    """
    Deterministic execution layer (The 'ACT' phase).

    Receives raw state_history (List[SimulationState]) plus the Analyst and
    Strategist outputs directly — the same pattern as apply_proportional_rule()
    and apply_human_intervener() in the industry baseline.

    Logic:
    - Bids:    Updated every virtual hour based on Strategist + Analyst signals.
    - Budgets: Not applied here; budget authority is delegated to Zupervisor only.
    - Safety:  Enforces absolute bid floors, market ceilings, and kill-switch.
    - Recovery: After exiting emergency mode, applies a jump-start floor (2.00 CHF)
               so bids immediately re-enter competitive territory.
    """

    # Recovery floor applied after any bid-pause (emergency or budget depletion).
    # Set at 4.50 CHF — above the competitive CPC threshold (~4.15 CHF) — so the
    # campaign re-enters competitive auctions immediately rather than crawling up
    # from 0.50 CHF over 32+ virtual hours.
    RECOVERY_BID_FLOOR: float = 4.20
    POST_HOLIDAY_STABILIZATION_HOURS: int = 24
    POST_HOLIDAY_MAX_UP_RAMP: float = 1.15
    HOLIDAY_CPA_SOFT_CUT: float = 135.0
    HOLIDAY_CPA_HARD_CUT: float = 160.0
    HOLIDAY_SOFT_THROTTLE_FACTOR: float = 0.55
    HOLIDAY_HARD_THROTTLE_FACTOR: float = 0.35
    HOLIDAY_SOFT_THROTTLE_FLOOR: float = 1.40
    HOLIDAY_HARD_THROTTLE_FLOOR: float = 1.00

    def __init__(
        self,
        policy_path: str = "agents/knowledge/policy_db.json",
        shared_kb_path: str = "agents/knowledge/shared_knowledge.json",
        analyst_log: str = "agents/logs/analyst_log.jsonl",
        strategist_log: str = "agents/logs/strategist_log.jsonl",
        log_path: str = "agents/logs/taskmaster_log.jsonl"
    ):
        self.policy_path = policy_path
        self.shared_kb_path = shared_kb_path
        self.analyst_log = analyst_log
        self.strategist_log = strategist_log
        self.log_path = log_path

        self.current_state = {
            "max_bid": 2.0,
            "daily_budget": 1000.0,
            "cycle_counter": 0,
        }
        # Emergency tracking: enables recovery floor after any bid-pause.
        self._was_emergency: bool = False
        # Countdown of OODA cycles during which the recovery floor stays active.
        # Set to 2 whenever a bid-pause ends (emergency OR budget depletion).
        # This ensures a short deterministic relaunch while avoiding prolonged
        # expensive over-bidding after recovery.
        self._recovery_cycles_remaining: int = 0

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _build_hourly_bid_schedule(
        self,
        old_bid: float,
        target_final_bid: float,
        n_hours: int,
        floor: float,
        ceiling: float,
        mode: str,
        decision: Dict[str, Any],
    ) -> List[float]:
        """
        One OODA call produces N hourly bids (applied by the simulator without extra LLM calls).
        Default: linear ramp from env bid to Taskmaster-final bid with per-hour step smoothing.
        """
        n_hours = max(1, int(n_hours))
        if mode in ("EMERGENCY", "EMERGENCY_OVERRIDE"):
            return [0.01] * n_hours

        raw_sched = decision.get("hourly_bid_schedule")
        if isinstance(raw_sched, list) and len(raw_sched) == n_hours:
            try:
                parsed = [float(x) for x in raw_sched]
                return [round(max(floor, min(b, ceiling)), 2) for b in parsed]
            except (TypeError, ValueError):
                pass

        if n_hours <= 1:
            return [round(max(floor, min(target_final_bid, ceiling)), 2)]

        ticks = [
            old_bid + (i / max(1, n_hours - 1)) * (target_final_bid - old_bid)
            for i in range(n_hours)
        ]
        seq: List[float] = [max(floor, min(ticks[0], ceiling))]
        for i in range(1, n_hours):
            lo, hi = seq[-1] * 0.82, seq[-1] * 1.18
            seq.append(max(floor, min(max(lo, min(ticks[i], hi)), ceiling)))
        seq[-1] = round(max(floor, min(target_final_bid, ceiling)), 2)
        return [round(x, 2) for x in seq]

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------

    def execute_cycle(
        self,
        state_history: List[SimulationState],
        analysis_result: Dict[str, Any],
        decision: Dict[str, Any],
        schedule_hours: int = 1,
    ) -> Dict[str, Any]:
        """
        Enforce bid guardrails and return final executable bid.

        Returns a report dict with:
          bid_execution.actual        → pass to env.configure(max_bid=...) for first hour of interval
          bid_schedule                → full hourly bids for this OODA interval (deterministic path)
          budget_execution.*          → telemetry only; no budget is applied here.
        """
        schedule_hours = max(1, int(schedule_hours))
        kb = self._load_json(self.shared_kb_path)
        latest = state_history[-1]

        self.current_state["cycle_counter"] += 1
        self.current_state["daily_budget"] = latest.biz_inputs.daily_budget

        current_budget = latest.biz_inputs.daily_budget
        old_bid = self.current_state["max_bid"]

        # ── 1. Extract Strategist's proposed bid and mode ───────────────────
        proposed_bid = decision.get("suggested_max_bid", self.current_state["max_bid"])
        mode         = decision.get("selected_mode", "NORMAL")
        safety_notes: List[str] = []
        current_hour_abs = int(latest.market_outcome.current_hour)
        in_holiday = HOLIDAY_HOURS[0] <= current_hour_abs <= HOLIDAY_HOURS[1]

        # ── 2. GUARDRAIL: Tech Kill-Switch ──────────────────────────────────
        # Use the Analyst's structured summary_signal (not fragile string matching).
        # TECH_FAILURE is only set when an authoritative tech_ping CRASH is active,
        # or when bids are competitive AND CVR has collapsed — never for bid-silence.
        is_tech_failure = analysis_result.get("summary_signal") == "TECH_FAILURE"
        if is_tech_failure and mode != "EMERGENCY":
            proposed_bid = 0.01
            mode = "EMERGENCY_OVERRIDE"
            safety_notes.append(
                "CRITICAL: Analyst confirmed TECH_FAILURE signal. Overriding to Kill-Switch."
            )

        # Relax CPA / budget throttles for all of holiday except emergency / tech outage.
        in_holiday_relaxed = (
            in_holiday
            and mode not in ("EMERGENCY", "EMERGENCY_OVERRIDE")
            and not is_tech_failure
        )

        # ── 2b Zero-lead 4h guard (F penalises spend with no conversions) ─────
        hist_4h = state_history[-16:]
        if hist_4h and mode not in ("EMERGENCY", "EMERGENCY_OVERRIDE"):
            sp4 = float(sum(s.market_outcome.spend for s in hist_4h))
            ld4 = float(sum(s.market_outcome.leads for s in hist_4h))
            if ld4 == 0.0 and sp4 >= 40.0:
                live_bid = float(latest.biz_inputs.max_bid)
                proposed_bid = min(float(proposed_bid), live_bid * 0.88)
                safety_notes.append(
                    f"Zero-lead 4h guard: spend {sp4:.2f} CHF, 0 leads → throttle proposed bid."
                )

        # ── 3. GUARDRAIL: Absolute Policy Bounds ────────────────────────────
        ceiling = kb.get("baseline_environment", {}).get("market_ceiling_cpc", 20.0)
        floor   = 0.50  # policy floor for Zurich Audi

        # Recovery floor: active on every cycle while _recovery_cycles_remaining > 0
        # OR on the first non-emergency cycle (_was_emergency still True).
        # Ensures at least 4 OODA cycles at competitive bid after any bid-pause,
        # preventing the sub-competitive crawl from 0.50 CHF.
        if mode not in ["EMERGENCY", "EMERGENCY_OVERRIDE"] and (
            self._was_emergency or self._recovery_cycles_remaining > 0
        ):
            floor = max(floor, self.RECOVERY_BID_FLOOR)
            safety_notes.append(
                f"Recovery floor active (cycles remaining: {self._recovery_cycles_remaining}): "
                f"bid floor raised to CHF {floor:.2f}."
            )

        final_bid = max(floor, min(proposed_bid, ceiling))

        # Confirmed emergencies bypass the floor
        if mode in ["EMERGENCY", "EMERGENCY_OVERRIDE"]:
            final_bid = 0.01

        # ── 4. Post-holiday stabilization guard ──────────────────────────────
        # The first 24h after holiday are volatile; cap upward ramps to avoid
        # overspend spikes and repeated stop/start behavior.
        holiday_end_hour = HOLIDAY_HOURS[1]
        in_post_holiday_stabilization = (
            holiday_end_hour < current_hour_abs <= holiday_end_hour + self.POST_HOLIDAY_STABILIZATION_HOURS
        )
        if in_post_holiday_stabilization and mode not in ["EMERGENCY", "EMERGENCY_OVERRIDE"]:
            max_allowed = old_bid * self.POST_HOLIDAY_MAX_UP_RAMP
            if final_bid > max_allowed:
                final_bid = max_allowed
                safety_notes.append(
                    "Post-holiday stabilization: capped upward bid ramp to +15% for smoother transition."
                )

        # ── 5. Efficiency guardrail: trim bids during sustained high CPA ───────
        history_24h = state_history[-96:]
        if mode not in ["EMERGENCY", "EMERGENCY_OVERRIDE"] and history_24h:
            spend_24h = float(sum(s.market_outcome.spend for s in history_24h))
            leads_24h = float(sum(s.market_outcome.leads for s in history_24h))
            clicks_24h = float(sum(s.market_outcome.clicks for s in history_24h))
            cpa_24h = (spend_24h / leads_24h) if leads_24h > 0 else float("inf")
            if clicks_24h >= 40:
                hard_cut = self.HOLIDAY_CPA_HARD_CUT if in_holiday_relaxed else 120.0
                soft_cut = self.HOLIDAY_CPA_SOFT_CUT if in_holiday_relaxed else 100.0
                if cpa_24h >= hard_cut:
                    final_bid = min(final_bid, max(0.50, old_bid * 0.80))
                    safety_notes.append(
                        f"Efficiency guardrail hard-cut: rolling 24h CPA={cpa_24h:.2f} >= {hard_cut:.2f}, capping bid to 80% of prior."
                    )
                elif cpa_24h >= soft_cut:
                    final_bid = min(final_bid, max(0.50, old_bid * 0.90))
                    safety_notes.append(
                        f"Efficiency guardrail soft-cut: rolling 24h CPA={cpa_24h:.2f} >= {soft_cut:.2f}, capping bid to 90% of prior."
                    )

        # ── 6. Budget safety: throttle before hard stop ──────────────────────
        # Replace unconditional pause with a two-tier throttle unless we are in
        # true emergency mode. This preserves auction participation while still
        # preventing runaway spend near budget depletion.
        is_budget_depleted = latest.derived_variables.budget_status == "budget_depleted"
        if is_budget_depleted and mode not in ["EMERGENCY", "EMERGENCY_OVERRIDE"]:
            day_spend = float(latest.derived_variables.current_day_spend or 0.0)
            budget_ratio = (day_spend / current_budget) if current_budget > 0 else 0.0
            if budget_ratio >= 1.05:
                if in_holiday_relaxed:
                    final_bid = min(
                        final_bid,
                        max(self.HOLIDAY_HARD_THROTTLE_FLOOR, old_bid * self.HOLIDAY_HARD_THROTTLE_FACTOR),
                    )
                    safety_notes.append(
                        "Holiday budget critically depleted — relaxed hard throttle "
                        f"({int(self.HOLIDAY_HARD_THROTTLE_FACTOR * 100)}% of prior, "
                        f"floor CHF {self.HOLIDAY_HARD_THROTTLE_FLOOR:.2f})."
                    )
                else:
                    final_bid = min(final_bid, max(0.50, old_bid * 0.15))
                    safety_notes.append(
                        "Daily budget critically depleted — applying hard throttle (15% of prior bid, floored at CHF 0.50)."
                    )
            else:
                if in_holiday_relaxed:
                    final_bid = min(
                        final_bid,
                        max(self.HOLIDAY_SOFT_THROTTLE_FLOOR, old_bid * self.HOLIDAY_SOFT_THROTTLE_FACTOR),
                    )
                    safety_notes.append(
                        "Holiday budget near depleted — relaxed soft throttle "
                        f"({int(self.HOLIDAY_SOFT_THROTTLE_FACTOR * 100)}% of prior, "
                        f"floor CHF {self.HOLIDAY_SOFT_THROTTLE_FLOOR:.2f})."
                    )
                else:
                    final_bid = min(final_bid, max(0.80, old_bid * 0.30))
                    safety_notes.append(
                        "Daily budget near depleted — applying soft throttle (30% of prior bid, floored at CHF 0.80)."
                    )
        elif is_budget_depleted:
            final_bid = 0.01
            safety_notes.append("Daily budget depleted during emergency — pausing bids until next calendar day.")

        live_anchor = float(latest.biz_inputs.max_bid)
        bid_schedule = self._build_hourly_bid_schedule(
            old_bid=live_anchor,
            target_final_bid=final_bid,
            n_hours=schedule_hours,
            floor=floor,
            ceiling=ceiling,
            mode=mode,
            decision=decision,
        )

        # ── 7. Update bid state (first hour of schedule) ───────────────────
        self.current_state["max_bid"] = round(float(bid_schedule[0]), 2)

        # Track whether a bid-pause just occurred (emergency or emergency-time
        # budget depletion). Non-emergency depletion now uses throttling.
        budget_depleted_pause = (
            is_budget_depleted
            and final_bid == 0.01
        )
        if mode in ["EMERGENCY", "EMERGENCY_OVERRIDE"] or budget_depleted_pause:
            self._was_emergency = True
            self._recovery_cycles_remaining = 2   # floor persists for 2 OODA cycles
        else:
            self._was_emergency = False
            if self._recovery_cycles_remaining > 0:
                self._recovery_cycles_remaining -= 1

        # ── 8. Budget telemetry only ────────────────────────────────────────
        # Budget authority lives in Zupervisor. Taskmaster records budget-related
        # context for auditability but does not apply budget changes.
        proposed_budget = float(decision.get("suggested_daily_budget", current_budget))
        final_budget = None
        budget_reason = "zupervisor_only_budget_authority"

        # ── 9. Audit log ────────────────────────────────────────────────────
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle": self.current_state["cycle_counter"],
            "virtual_day":  latest.market_outcome.current_day,
            "virtual_hour": latest.market_outcome.current_hour,
            "mode_executed": mode,
            "analyst_signal": analysis_result.get("summary_signal", "UNKNOWN"),
            "bid_execution": {
                "proposed": round(proposed_bid, 2),
                "actual":   self.current_state["max_bid"],
                "delta":    round(self.current_state["max_bid"] - old_bid, 2),
            },
            "bid_schedule": bid_schedule,
            "budget_execution": {
                "final_budget":         final_budget,
                "proposed":             round(proposed_budget, 2),
                "current":              round(current_budget, 2),
                "hours_since_last_change": None,
                "reason":               budget_reason,
            },
            "safety_overrides": safety_notes if safety_notes else [],
        }

        self._log_execution(report)
        return report

    def _log_execution(self, report: Dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(report) + "\n")
