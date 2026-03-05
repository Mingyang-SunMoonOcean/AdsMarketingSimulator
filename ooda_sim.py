"""Entry point to run Phase 2 OODA MAS simulation."""

from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional, Tuple

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState
from core.volatility_scheduler import VolatilityScheduler, HOLIDAY_HOURS
from agents.analyst import Analyst
from agents.strategist import Strategist
from agents.taskmaster import Taskmaster
from agents.zupervisor import Zupervisor


# ---------------------------------------------------------------------------
# Configuration (mirror Industry Baseline for comparability)
# ---------------------------------------------------------------------------
STEP_MINUTES = 15
HOURS_PER_DAY = 24
STEPS_PER_HOUR = int(60 / STEP_MINUTES)          # 4
TOTAL_DAYS = 37
TOTAL_STEPS = TOTAL_DAYS * HOURS_PER_DAY * STEPS_PER_HOUR   # 3,552
EFFECTIVE_STEPS = 30 * HOURS_PER_DAY * STEPS_PER_HOUR       # 2,880

INITIAL_MAX_BID = 5.00
INITIAL_DAILY_BUDGET = 1000.0

# Control loop frequencies (mirrors industry_baseline_sim structure)
#
# Three OODA cadences are used depending on market regime:
#
#   NORMAL (every 4 virtual hours):
#       Covers 888 virtual hours of baseline operation → 222 OODA cycles, 75 %
#       fewer LLM calls than the original 1-hour cadence.  The Taskmaster's 24 h
#       budget-gate means finer resolution adds no decision value.
#
#   HOLIDAY (every 2 virtual hours):
#       CVR is +30 % during the surge window; more frequent cycles let the
#       Strategist ramp bids faster and fully exploit the elevated conversion rate.
#       156 holiday hours → 78 cycles (vs 39 at normal cadence).
#
#   CRASH / RECOVERY_PENDING (suppressed):
#       Once the initial kill-switch cycle fires (bid → 0.01), the campaign is
#       already paused.  No further OODA cycles are needed until the RECOVERY
#       ping arrives.  Without this suppression the loop fires every 15 virtual
#       minutes during the outage because TechMonitor emits a CRASH ping on
#       every step while CRASH_ACTIVE — wasting ~576 LLM calls for a 6-day crash.
#       OODA fires exactly twice for the whole outage: once on crash activation,
#       once on confirmed recovery.
OODA_STEP_INTERVAL          = 4 * STEPS_PER_HOUR              # every 4 virtual hours (normal)
HOLIDAY_OODA_STEP_INTERVAL  = 2 * STEPS_PER_HOUR             # every 2 virtual hours (holiday)
POST_HOLIDAY_OODA_STEP_INTERVAL = 2 * STEPS_PER_HOUR         # temporary 2h cadence after holiday
POST_HOLIDAY_STABILIZATION_HOURS = 24                        # stabilization window length
ZUPERVISOR_INTERVAL         = 5 * HOURS_PER_DAY * STEPS_PER_HOUR  # every 5 virtual days
HOLIDAY_ZUPERVISOR_INTERVAL = HOURS_PER_DAY * STEPS_PER_HOUR     # every 1 virtual day during holiday

# Agent knowledge / log paths — absolute so agents work from any CWD
_AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
SHARED_KNOWLEDGE_PATH = os.path.join(_AGENTS_DIR, "knowledge", "shared_knowledge.json")
POLICY_PATH = os.path.join(_AGENTS_DIR, "knowledge", "policy_db.json")
ANALYST_LOG_PATH = os.path.join(_AGENTS_DIR, "logs", "analyst_log.jsonl")
STRATEGIST_LOG_PATH = os.path.join(_AGENTS_DIR, "logs", "strategist_log.jsonl")
TASKMASTER_LOG_PATH = os.path.join(_AGENTS_DIR, "logs", "taskmaster_log.jsonl")
ZUPERVISOR_LOG_PATH = os.path.join(_AGENTS_DIR, "logs", "zupervisor_log.jsonl")


# ---------------------------------------------------------------------------
# TechMonitor — simulates tech-department pings to the MAS
# ---------------------------------------------------------------------------

class TechMonitor:
    """
    Simulates the tech department's real-time alerting channel to the MAS.

    Mirrors the event-based alerting logic in baseline/legacy_human.py:
      - 1-hour (4-step) lag before a CRASH alert is issued
      - 30-minute (2-step) lag before a RECOVERY alert is confirmed

    The holiday period is NOT handled here — it is a KNOWN event that lives
    in the shared_knowledge event_calendar and is anticipated by agents from
    the calendar alone. The crash is UNPREDICTABLE and is communicated only
    via this ping mechanism.

    State machine:
        NORMAL → (crash detected) → CRASH_PENDING → (lag met) → CRASH_ACTIVE
        CRASH_ACTIVE → (crash ends) → RECOVERY_PENDING → (lag met) → NORMAL
    """

    CRASH_DETECTION_LAG: int = 1 * STEPS_PER_HOUR       # 1 virtual hour  = 4 steps
    RECOVERY_CONFIRM_LAG: int = STEPS_PER_HOUR // 2     # 30 virtual mins = 2 steps

    def __init__(self) -> None:
        self._state: str = "NORMAL"
        self._crash_onset_step: Optional[int] = None
        self._recovery_onset_step: Optional[int] = None

    def check(self, step: int, scheduler_event: str) -> Optional[Dict[str, Any]]:
        """
        Call every simulation step with the current scheduler event string.

        Returns a tech_ping dict when an alert is active, or None otherwise.
        The Analyst receives this directly via tech_ping= — it is treated as
        an authoritative external signal, not inferred from market data.
        """
        if scheduler_event == "CRASH":
            # Transition: NORMAL → CRASH_PENDING on first sighting
            if self._state == "NORMAL":
                self._state = "CRASH_PENDING"
                self._crash_onset_step = step
                self._recovery_onset_step = None

            # Transition: CRASH_PENDING → CRASH_ACTIVE after detection lag
            if (self._state == "CRASH_PENDING"
                    and step >= self._crash_onset_step + self.CRASH_DETECTION_LAG):
                self._state = "CRASH_ACTIVE"

            # Active crash: return ping every cycle until event resolves
            if self._state == "CRASH_ACTIVE":
                return {
                    "event": "CRASH",
                    "detected_at_step": self._crash_onset_step + self.CRASH_DETECTION_LAG,
                    "message": (
                        "TECH DEPT ALERT: Website outage confirmed. "
                        "CVR is near-zero — all spend is toxic leakage. "
                        "Immediate bid pause required. Do not resume until RECOVERY is confirmed."
                    ),
                    "severity": "CRITICAL",
                    "source": "tech_department",
                }

        else:
            # Transition: CRASH_* → RECOVERY_PENDING once crash ends
            if self._state in ("CRASH_ACTIVE", "CRASH_PENDING"):
                self._state = "RECOVERY_PENDING"
                self._recovery_onset_step = step
                self._crash_onset_step = None

            # Transition: RECOVERY_PENDING → NORMAL after confirmation lag
            if (self._state == "RECOVERY_PENDING"
                    and step >= self._recovery_onset_step + self.RECOVERY_CONFIRM_LAG):
                self._state = "NORMAL"
                self._recovery_onset_step = None
                return {
                    "event": "RECOVERY",
                    "message": (
                        "TECH DEPT ALERT: Website recovery confirmed. "
                        "CVR is restored to baseline. "
                        "Safe to resume bidding — verify CVR in recent data before scaling."
                    ),
                    "severity": "INFO",
                    "source": "tech_department",
                }

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_log_files() -> None:
    """Truncate all agent JSONL logs so each simulation run starts clean."""
    for path in [ANALYST_LOG_PATH, STRATEGIST_LOG_PATH,
                 TASKMASTER_LOG_PATH, ZUPERVISOR_LOG_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()  # truncate


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_ooda_simulation(
    total_steps: int = TOTAL_STEPS,
    initial_max_bid: float = INITIAL_MAX_BID,
    initial_daily_budget: float = INITIAL_DAILY_BUDGET,
) -> Tuple[List[SimulationState], dict]:
    """
    Run the OODA MAS simulation (Phase 2).

    External signal routing:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Holiday calendar  → shared_knowledge.json (event_calendar section) │
    │                      agents anticipate it from knowledge alone.      │
    │                                                                      │
    │  Website crash     → TechMonitor (mimics tech dept ping channel)    │
    │                      4-hour detection lag, 2-hour recovery lag.      │
    │                      Delivered to Analyst via tech_ping= parameter.  │
    └─────────────────────────────────────────────────────────────────────┘

    All agents receive state_history directly from env.observe() exactly as
    apply_proportional_rule() and apply_human_intervener() do in the baseline.

    Two control loops:

    Loop A — OODA cycle (every virtual hour, i.e. every 4 steps):
        Observe  : analyst.analyze(state_history, tech_ping)
        Decide   : strategist.decide(analysis_result, state_history)
        Act      : taskmaster.execute_cycle(state_history, analysis_result, decision)
                   → env.configure(max_bid=final_bid)

    Loop B — Zupervisor review (every 5 virtual days):
        zupervisor.provide_guidance(current_day, state_history)
        strategist.reload_policy()   ← picks up updated alpha/beta

    Returns a tuple of (effective_state_history, aggregate_metrics).
    """
    _clear_log_files()

    scheduler = VolatilityScheduler()
    env = SandboxEnv(scheduler=scheduler)
    tech_monitor = TechMonitor()

    analyst = Analyst(
        shared_knowledge_path=SHARED_KNOWLEDGE_PATH,
        policy_path=POLICY_PATH,
        log_path=ANALYST_LOG_PATH,
    )
    strategist = Strategist(
        shared_knowledge_path=SHARED_KNOWLEDGE_PATH,
        policy_path=POLICY_PATH,
        log_path=STRATEGIST_LOG_PATH,
    )
    taskmaster = Taskmaster(
        policy_path=POLICY_PATH,
        shared_kb_path=SHARED_KNOWLEDGE_PATH,
        log_path=TASKMASTER_LOG_PATH,
    )
    zupervisor = Zupervisor(
        policy_path=POLICY_PATH,
        shared_kb_path=SHARED_KNOWLEDGE_PATH,
        log_path=ZUPERVISOR_LOG_PATH,
    )

    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)

    # Persist the latest tech ping across steps so the Analyst always sees the
    # most current alert (CRASH stays active; RECOVERY clears it after one cycle).
    active_tech_ping: Optional[Dict[str, Any]] = None
    # Track TechMonitor state across steps to detect one-shot transitions
    # (CRASH_ACTIVE entry and RECOVERY confirmation) without re-firing on every step.
    _prev_tech_state: str = "NORMAL"

    for step in range(total_steps):
        # Advance the simulation by one 15-minute tick
        # Print current step number, virtual hour, and virtual day
        print(f"Step {step}, Virtual Hour {env.clock.current_hour}, Virtual Day {env.clock.current_day}")
        env.act()
        state_history = env.observe()

        # ------------------------------------------------------------------
        # Tech monitoring — checked every step for precise lag tracking.
        # Holiday calendar is in shared_knowledge; only crashes go via ping.
        # ------------------------------------------------------------------
        current_hour = env.clock.current_hour
        sched_event = scheduler.get_v_multiplier(current_hour)["event"]
        new_ping = tech_monitor.check(step, sched_event)

        # ------------------------------------------------------------------
        # Budget control hook (Zupervisor-owned, baseline-equivalent triggers)
        # Checked each step for trigger parity while keeping strategic review
        # cadence unchanged.
        # ------------------------------------------------------------------
        budget_report = zupervisor.evaluate_budget_triggers(
            current_hour=current_hour,
            scheduler_event=sched_event,
            state_history=state_history,
        )
        if budget_report is not None:
            env.configure(daily_budget=budget_report["suggested_daily_budget"])
            suggested_max_bid = budget_report.get("suggested_max_bid")
            if suggested_max_bid is not None:
                env.configure(max_bid=float(suggested_max_bid))

        if new_ping is not None:
            # CRASH ping stays active until recovery; RECOVERY clears after one cycle.
            active_tech_ping = new_ping
        elif tech_monitor._state == "NORMAL" and (
            active_tech_ping is not None
            and active_tech_ping.get("event") == "RECOVERY"
        ):
            # Clear after the RECOVERY ping has been delivered once
            active_tech_ping = None

        curr_tech_state = tech_monitor._state

        # ── One-shot transition triggers ──────────────────────────────────
        # CRASH: fire OODA exactly once when crash first becomes confirmed
        # (TechMonitor just moved into CRASH_ACTIVE this step).
        is_crash_newly_active = (
            curr_tech_state == "CRASH_ACTIVE"
            and _prev_tech_state != "CRASH_ACTIVE"
        )
        # RECOVERY: fire OODA exactly once when recovery is confirmed
        # (TechMonitor emits a RECOVERY ping exactly once on transition).
        is_recovery_confirmed = (
            new_ping is not None and new_ping.get("event") == "RECOVERY"
        )
        _prev_tech_state = curr_tech_state

        # ── Cadence selection ─────────────────────────────────────────────
        # Suppress regular OODA ticks while the outage is ongoing:
        #   CRASH_ACTIVE      → bid is already at 0.01; nothing to decide.
        #   RECOVERY_PENDING  → crash ended but not confirmed; active_tech_ping
        #                       still shows CRASH, so Analyst output is identical.
        # Only the two one-shot triggers above fire during this window.
        in_disruption = curr_tech_state in ("CRASH_ACTIVE", "RECOVERY_PENDING")

        # During holiday surge use faster cadence; keep 24h higher cadence right
        # after holiday to stabilize transition back to normal regime.
        holiday_end_hour = HOLIDAY_HOURS[1]
        in_post_holiday_stabilization = (
            sched_event == "NORMAL"
            and holiday_end_hour < current_hour <= holiday_end_hour + POST_HOLIDAY_STABILIZATION_HOURS
        )
        effective_interval = (
            HOLIDAY_OODA_STEP_INTERVAL if sched_event == "HOLIDAY"
            else POST_HOLIDAY_OODA_STEP_INTERVAL if in_post_holiday_stabilization
            else OODA_STEP_INTERVAL
        )
        regular_ooda_tick = (
            not in_disruption
            and (step + 1) % effective_interval == 0
        )

        # ------------------------------------------------------------------
        # Loop A: OODA cycle
        #   • Normal regime  → every 4 virtual hours
        #   • Holiday regime → every 2 virtual hours
        #   • Crash/recovery → suppressed; only one-shot triggers fire
        # All agents receive state_history directly from env.observe().
        # ------------------------------------------------------------------
        if (
            (regular_ooda_tick or is_crash_newly_active or is_recovery_confirmed)
            and len(state_history) >= STEPS_PER_HOUR
        ):

            # Observe — Analyst interprets state_history + optional tech ping
            analysis_result = analyst.analyze(state_history, tech_ping=active_tech_ping)

            # Attach virtual simulation hour so Strategist's shadow price
            # λ = exp(t/24) tracks simulation time, not real wall-clock UTC.
            virtual_hour = env.clock.current_hour % 24
            analysis_result["timestamp"] = f"2000-01-01T{virtual_hour:02d}:00:00"

            # Orient + Decide — Strategist reads current bid/budget from state_history
            decision = strategist.decide(analysis_result, state_history)

            # Act — Taskmaster enforces guardrails, receives everything directly
            execution = taskmaster.execute_cycle(state_history, analysis_result, decision)
            env.configure(max_bid=execution["bid_execution"]["actual"])

            # Clear one-shot RECOVERY ping after it has been processed
            if (active_tech_ping is not None
                    and active_tech_ping.get("event") == "RECOVERY"):
                active_tech_ping = None

        # ------------------------------------------------------------------
        # Loop B: Zupervisor strategic review — every 5 virtual days,
        # or daily during the holiday window (mirrors IB: human supervisor
        # checks in every day during a known high-stakes surge).
        # Receives state_history directly, mirrors apply_human_intervener().
        # ------------------------------------------------------------------
        regular_zuper_tick = (step + 1) % ZUPERVISOR_INTERVAL == 0
        holiday_zuper_tick = (
            sched_event == "HOLIDAY"
            and (step + 1) % HOLIDAY_ZUPERVISOR_INTERVAL == 0
        )
        if (regular_zuper_tick or holiday_zuper_tick) and len(state_history) >= ZUPERVISOR_INTERVAL:
            current_day = env.clock.current_day
            # Use 1-day window for holiday-only ticks; full 5-day window for regular.
            window = (
                state_history[-HOLIDAY_ZUPERVISOR_INTERVAL:]
                if holiday_zuper_tick and not regular_zuper_tick
                else state_history[-ZUPERVISOR_INTERVAL:]
            )
            guidance = zupervisor.provide_guidance(current_day, window)
            strategist.reload_policy()

            print(
                f"[Zupervisor] Day {current_day}: "
                f"{guidance.get('action')} — {guidance.get('reasoning')}"
            )

    state_history = env.observe()
    effective_state_history = state_history[-EFFECTIVE_STEPS:]
    total_spend = sum(s.market_outcome.spend for s in effective_state_history)
    total_leads = sum(s.market_outcome.leads for s in effective_state_history)
    total_clicks = sum(s.market_outcome.clicks for s in effective_state_history)
    aggregate_metrics = {
        "total_spend": round(total_spend, 4),
        "total_leads": int(total_leads),
        "total_clicks": int(total_clicks),
        "overall_cpa": round(total_spend / total_leads, 4) if total_leads > 0 else None,
    }

    return effective_state_history, aggregate_metrics


# ---------------------------------------------------------------------------
# CSV output — identical schema to ib_results.csv for side-by-side comparison
# ---------------------------------------------------------------------------

def write_to_csv(results_csv_path: str, state_history: List[SimulationState]) -> None:
    """Write state history to CSV in the same schema as ib_results.csv."""
    os.makedirs(os.path.dirname(results_csv_path) or ".", exist_ok=True)
    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    if not state_history:
        return

    fieldnames = [
        "current_day", "current_hour", "current_minute",
        "clicks", "leads", "spend", "cpa", "realized_cpc",
        "daily_budget", "max_bid", "volatility",
        "budget_status", "current_day_spend",
    ]

    with open(results_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in state_history:
            writer.writerow({
                "current_day": s.market_outcome.current_day,
                "current_hour": s.market_outcome.current_hour,
                "current_minute": s.market_outcome.current_minute,
                "clicks": s.market_outcome.clicks,
                "leads": s.market_outcome.leads,
                "spend": s.market_outcome.spend,
                "cpa": s.market_outcome.cpa,
                "realized_cpc": s.market_outcome.realized_cpc,
                "daily_budget": s.biz_inputs.daily_budget,
                "max_bid": s.biz_inputs.max_bid,
                "volatility": s.external_events_inputs.volatility,
                "budget_status": s.derived_variables.budget_status,
                "current_day_spend": s.derived_variables.current_day_spend,
            })


if __name__ == "__main__":
    effective_state_history, metrics = run_ooda_simulation()

    write_to_csv("data/mas_results.csv", effective_state_history)

    latest = effective_state_history[-1] if effective_state_history else None
    print("=== OODA MAS Simulation (Phase 2, 30 days) ===")
    if latest:
        print(f"Final virtual day: {latest.market_outcome.current_day}")
        print(f"Final max_bid: {latest.biz_inputs.max_bid}")
        print(f"Final daily_budget: {latest.biz_inputs.daily_budget}")
    print("--- Aggregates ---")
    print(f"Total spend: {metrics['total_spend']}")
    print(f"Total leads: {metrics['total_leads']}")
    print(f"Total clicks: {metrics['total_clicks']}")
    print(f"Overall CPA: {metrics['overall_cpa']}")
