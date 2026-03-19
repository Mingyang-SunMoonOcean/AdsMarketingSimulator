"""Observation / Anomaly Detection agent for OODA MAS."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from core.state_manager import SimulationState

# Retry config for transient OpenAI errors
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 5.0   # initial backoff; doubles each attempt, capped at 60 s
_API_TIMEOUT_S = 90.0  # per-request timeout passed to the SDK


class Analyst:
    """
    Analyst agent: Observation and Anomaly Detection.

    Receives the raw state_history list (same format SandboxEnv.observe() returns),
    mirrors the interface of baseline/rule_engine.py and baseline/legacy_human.py.

    This agent:
    - Interprets recent performance context from state_history
    - Uses shared knowledge + policy grounding
    - Produces structured analytical output
    - Does NOT make decisions
    """

    def __init__(
        self,
        shared_knowledge_path: str,
        policy_path: str,
        log_path: str = "agents/logs/analyst_log.jsonl",
        model: str = "gpt-4.1-nano"
    ):
        load_dotenv()
        self.client = OpenAI()
        self.model = model

        self.shared_knowledge = self._load_json(shared_knowledge_path)
        self.policy = self._load_json(policy_path)
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Build once — static knowledge/policy is embedded so OpenAI's prompt
        # caching can reuse the identical prefix across all calls this run.
        self._system_prompt = self._build_system_prompt()

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # State History → DataFrame conversion
    # (mirrors what rule_engine.py does with state_history)
    # --------------------------------------------------

    @staticmethod
    def _to_dataframe(state_history: List[SimulationState]) -> pd.DataFrame:
        """Convert raw state_history (from env.observe()) to a pandas DataFrame."""
        return pd.DataFrame([
            {
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
            }
            for s in state_history
        ])

    # --------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------

    def analyze(
        self,
        state_history: List[SimulationState],
        tech_ping: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Observe and interpret the simulation state.

        Accepts raw state_history (List[SimulationState]) exactly as returned by
        env.observe() — the same interface used by apply_proportional_rule() and
        apply_human_intervener() in the industry baseline.
        """
        historical_df = self._to_dataframe(state_history)

        # Recent context windows
        window_df = historical_df.tail(4)     # last 1 hour  (4 steps)
        loopback_df = historical_df.tail(16)  # last 4 hours (16 steps)
        rolling_24h_lookback_df = historical_df.tail(96)  # last 24 hours (96 steps)

        # Only dynamic per-call data goes in the user message; static knowledge/policy
        # lives in self._system_prompt (built once in __init__) so it can be cached.
        payload = {
            "recent_window_data": window_df.to_dict(orient="records"),
            "loopback_context": loopback_df.to_dict(orient="records"),
            "rolling_24h_lookback_context": rolling_24h_lookback_df.to_dict(orient="records"),
            "technology_ping": tech_ping,
        }

        user_prompt = json.dumps(payload)

        raw = self._call_llm([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        analysis_output = json.loads(raw)
        self._log_analysis(analysis_output)
        return analysis_output

    # --------------------------------------------------
    # LLM call with timeout + retry
    # --------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call OpenAI with a hard per-request timeout and exponential-backoff
        retry on transient errors (rate limit, timeout, connection drop).

        Raises the underlying exception if all attempts are exhausted.
        """
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=messages,
                    timeout=_API_TIMEOUT_S,
                )
                return response.choices[0].message.content
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise
                delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), 60.0)
                print(
                    f"[Analyst] {type(exc).__name__} on attempt {attempt}/{_MAX_ATTEMPTS}. "
                    f"Retrying in {delay:.0f}s…",
                    flush=True,
                )
                time.sleep(delay)

    # --------------------------------------------------
    # Prompt Construction
    # --------------------------------------------------

    def _build_system_prompt(self) -> str:
        return """
You are the Analyst Agent in a Multi-Agent OODA system for a Zurich Audi dealership campaign.

Your role:
- Observe performance metrics from state_history.
- Contextualize them using shared knowledge, event_calendar, and policy.
- Detect patterns, anomalies, and regimes.
- Assess efficiency, leakage risk, pacing status, and technology impact.
- DO NOT suggest actions. DO NOT make decisions. Only analyze and interpret.

---

CRITICAL — DISTINGUISHING SELF-INFLICTED BID SILENCE FROM REAL MARKET FAILURES:

Before classifying near-zero activity (zero leads, zero spend, near-zero CVR) as
TECH_FAILURE, LEAKAGE_RISK, or NEGATIVE_SHOCK, you MUST inspect the max_bid field
in recent_window_data:

  LOW-BID SILENCE RULE:
  If ALL max_bid values in recent_window_data are < 4.20 CHF:
    - The reduced or near-zero activity is caused by our own sub-market bid, NOT a website
      failure, leakage, or negative shock.
    - The market competitive CPC threshold is ~4.15 CHF. Bids below this threshold win
      proportionally fewer auctions:
        • bid < 0.80 CHF  → almost zero auction wins ("bid-floor silence")
        • bid 0.80–4.15 CHF → partial auction volume, proportionally reduced activity
        • bid ≥ 4.15 CHF  → full competitive participation
    - This sub-competitive state arises during bid recovery after any pause (emergency,
      budget depletion, or crash recovery). It is NOT a structural market failure.
    - Set technology_impact: "No technical outages detected. Market activity is reduced
      by sub-competitive max_bid (< 4.20 CHF vs competitive threshold ~4.15 CHF).
      Bid recovery is in progress — this is NOT a market or tech failure."
    - Set summary_signal = "EFFICIENCY_DRIFT" (system is recovering; normal operation
      expected once bids reach competitive levels).
    - Do NOT set summary_signal = "TECH_FAILURE", "LEAKAGE_RISK", or "NEGATIVE_SHOCK"
      solely because activity is low when bids are sub-competitive.

  TECH_FAILURE may only be set when:
    (a) technology_ping.event == "CRASH" is present (authoritative), OR
    (b) technology_ping is null AND max_bid ≥ 4.20 CHF in recent_window_data
        AND the CVR (leads ÷ clicks where clicks > 0) is < 0.005.
    Sub-competitive bids produce reduced activity identical to market downturns or outages
    in the raw metrics — you MUST check bid level before inferring any external failure.

EFFICIENCY ESCALATION RULE (MANDATORY):
Use rolling_24h_lookback_context to compute:
  - rolling_24h_cpa = total_spend / max(total_leads, 1)
  - rolling_24h_clicks = total_clicks
If ALL conditions hold:
  1) rolling_24h_clicks >= 40
  2) rolling_24h_cpa >= 120 CHF
  3) at least one max_bid in recent_window_data is >= 4.20 CHF
then summary_signal MUST be "LEAKAGE_RISK" (not EFFICIENCY_DRIFT).
This prevents prolonged high-bid, high-CPA drift from being misclassified as benign noise.

---

HOW TO USE EXTERNAL SIGNALS:

1. TECHNOLOGY PING (technology_ping field in this message):
   - This is a real-time alert from the tech department. Treat it as AUTHORITATIVE.
   - If technology_ping.event == "CRASH": The website is confirmed down. CVR is near-zero.
     Set technology_impact to clearly state an outage is active and toxic spend is occurring.
     Set summary_signal = "TECH_FAILURE" regardless of what the market data shows.
   - If technology_ping.event == "RECOVERY": The website has been confirmed restored.
     Verify CVR in recent state_history before concluding the market has normalised.
     Set technology_impact to reflect recovery is confirmed but CVR needs verification.
   - If technology_ping is null: No tech alert active. Use market data to infer tech status.
     State "No technical outages detected" in technology_impact if data looks normal.

2. EVENT CALENDAR (event_calendar in shared_knowledge):
   - Contains KNOWN future events the system can anticipate (e.g., holiday period).
   - Cross-reference the current_day from state_history with event virtual_day_start/end.
   - If current_day is within 1 day BEFORE an event: flag it in market_context_summary
     as an anticipatory warning ("Holiday period begins tomorrow — pre-emptive adjustment warranted").
   - If current_day is WITHIN an event window: reference expected_effect in your assessment.
   - The website crash event is NOT in the calendar in advance — it arrives ONLY via tech_ping.

---

HOW TO USE SHARED KNOWLEDGE:
- baseline_environment: Use target_cpa, baseline_cvr, baseline_cpc, market_ceiling_cpc to assess deviations.
- market_patterns: Match observed behavior to patterns (holiday_surge, landing_page_failure, competitive_bid_spike, normal_noise).
- historical_incidents: Reference lessons learned to avoid misclassification.
- volatility_regimes: Classify into stable, negative_shock, or positive_surge using cvr_multiplier_range and known information (website crash, holiday period, etc.).
- efficiency_relationships: Apply leakage_condition (CPA > target AND CVR < baseline = toxic spend).
- noise_model_assumptions: Require confirmation before treating single-hour moves as structural.
- utility_intuition: Interpret metrics in terms of lead_value, cpa_penalty_weight, leakage_penalty_weight.

HOW TO USE POLICY CONTEXT:
- optimization_objective: F = U + Pacing Score; understand volume_value, cpa_penalty, leakage_penalty, no_conversion_penalty, and pacing reward/overspend penalty.
- cpa_efficiency_policy: Target CPA 80, acceptable band, hard threshold 120.
- budget_pacing_policy: Overspend/underspend tolerances; late_day_priority_increase.
- volatility_response_policy: CVR drop thresholds (moderate 40%, severe 70%); positive_surge_multiplier.
- risk_appetite_policy: max_bid_change_per_hour_percent; prefer gradual adjustments.
- escalation_policy: Flag when conditions_for_escalation are met (CPA>120 for 3h, pause>6h, budget deviation>20%).
- noise_protection_policy: minimum_confirmation_hours; do not react to single-hour anomalies.

---

You must output STRICTLY valid JSON.

Required output fields:

{
  "analysis_result": {
    "volatility_regime": "...",
    "efficiency_assessment": "...",
    "leakage_risk": "...",
    "budget_pacing_status": "...",
    "technology_impact": "...",
    "market_context_summary": "..."
  },
  "reasoning": {
    "metric_interpretation": "...",
    "knowledge_reference": "...",
    "policy_reference": "...",
    "event_calendar_reference": "...",
    "tech_ping_interpretation": "...",
    "uncertainty_notes": "..."
  },
  "confidence_score": 0.0-1.0,
  "summary_signal": "ONE_OF: STABLE | NEGATIVE_SHOCK | POSITIVE_SURGE | LEAKAGE_RISK | TECH_FAILURE | EFFICIENCY_DRIFT"
}

Guidelines:
- technology_ping takes PRIORITY over market data inference for technology_impact.
- event_calendar cross-reference is MANDATORY — always check current_day vs event windows.
- Use shared knowledge to classify regime and match market patterns.
- Distinguish noise from structural change using noise_model_assumptions.
- Be economically rational and remain neutral and analytical.

---

SHARED_KNOWLEDGE:
""" + json.dumps(self._extract_relevant_knowledge()) + """

POLICY_CONTEXT:
""" + json.dumps(self._extract_relevant_policy()) + """
"""

    def _extract_relevant_knowledge(self) -> Dict[str, Any]:
        return {
            "baseline_environment": self.shared_knowledge.get("baseline_environment"),
            "market_patterns": self.shared_knowledge.get("market_patterns"),
            "historical_incidents": self.shared_knowledge.get("historical_incidents"),
            "efficiency_relationships": self.shared_knowledge.get("efficiency_relationships"),
            "utility_intuition": self.shared_knowledge.get("utility_intuition"),
            "volatility_regimes": self.shared_knowledge.get("volatility_regimes"),
            "noise_model_assumptions": self.shared_knowledge.get("noise_model_assumptions"),
            "stakeholder_expectations": self.shared_knowledge.get("stakeholder_expectations"),
            # Holiday calendar and event schedule — agents use this for anticipatory reasoning
            "event_calendar": self.shared_knowledge.get("event_calendar"),
        }

    def _extract_relevant_policy(self) -> Dict[str, Any]:
        return {
            "optimization_objective": self.policy.get("optimization_objective"),
            "cpa_efficiency_policy": self.policy.get("cpa_efficiency_policy"),
            "budget_pacing_policy": self.policy.get("budget_pacing_policy"),
            "volatility_response_policy": self.policy.get("volatility_response_policy"),
            "risk_appetite_policy": self.policy.get("risk_appetite_policy"),
            "escalation_policy": self.policy.get("escalation_policy"),
            "noise_protection_policy": self.policy.get("noise_protection_policy"),
        }

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log_analysis(self, analysis: Dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(analysis) + "\n")
