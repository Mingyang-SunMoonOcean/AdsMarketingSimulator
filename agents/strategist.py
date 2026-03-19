"""Policy selection & Shadow Pricing (exp(t/24)) agent for OODA MAS."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import pandas as pd

from core.state_manager import SimulationState

# Retry config for transient OpenAI errors
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 5.0   # initial backoff; doubles each attempt, capped at 60 s
_API_TIMEOUT_S = 90.0  # per-request timeout passed to the SDK

BID_FLOOR = 0.50       # CHF – absolute minimum bid (mirrors rule_engine)
BID_CEILING = 20.00    # CHF – absolute maximum bid (mirrors rule_engine)
MAX_BID_CHANGE_PCT = 0.20   # ±20 % per decision cycle for smoother control
INITIAL_MAX_BID = 5.00      # Baseline human reset bid
LOW_UTIL_CHECK_INTERVAL_HOURS = 12
LOOKBACK_STEPS_24H = 96
MIN_GUARDRAIL_CLICKS = 25
NORMAL_MODE_MAX_UP = 1.12
NORMAL_MODE_MAX_DN = 0.82
OPPORTUNITY_MODE_MAX_UP = 1.20
CPA_SOFT_CUT = 100.0
CPA_HARD_CUT = 120.0


class Strategist:
    """
    Strategist agent: Policy Selection and Shadow Pricing.

    Consumes the Analyst's structured signal, consults the policy_db and
    shared_knowledge_db, and uses an LLM (GPT) to select one of three
    operating modes:

      NORMAL      – Standard shadow pricing; F = U + Pacing Score governs bid adjustments.
      EMERGENCY   – Leakage / tech failure / severe CVR shock; retreat toward bid floor.
      OPPORTUNITY – Confirmed positive CVR surge; aggressive but bounded scaling.

    Enforces all policy constraints programmatically after LLM reasoning and
    appends a fully-reasoned JSON decision to strategist_log.jsonl so that
    the Taskmaster (Executor) can read it.
    """

    def __init__(
        self,
        shared_knowledge_path: str,
        policy_path: str,
        log_path: str = "logs/strategist_log.jsonl",
        model: str = "gpt-4.1-nano",
    ):
        load_dotenv()
        self.client = OpenAI()
        self.model = model

        self.policy_path = policy_path
        self.shared_knowledge = self._load_json(shared_knowledge_path)
        self.policy = self._load_json(policy_path)
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Build once — static policy/knowledge is embedded so OpenAI's prompt
        # caching can reuse the identical prefix across all calls this run.
        # Rebuilt in reload_policy() when the Zupervisor updates policy_db.json.
        self._system_prompt = self._build_system_prompt()

    # --------------------------------------------------
    # JSON Loader
    # --------------------------------------------------

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------

    def decide(
        self,
        analysis_result: Dict[str, Any],
        state_history: List[SimulationState],
    ) -> Dict[str, Any]:
        """
        Produce a strategic bidding decision from the Analyst's signal.

        Accepts raw state_history (List[SimulationState]) exactly as returned by
        env.observe() — the same interface used by apply_proportional_rule() and
        apply_human_intervener() in the industry baseline.

        Parameters
        ----------
        analysis_result : dict
            Full output from Analyst.analyze() — contains analysis_result,
            reasoning, confidence_score, summary_signal, and timestamp.
        state_history : List[SimulationState]
            Raw state history from env.observe(); latest entry used for current
            max_bid and daily_budget.

        Returns
        -------
        dict
            selected_mode, selected_policies, bid_multiplier,
            suggested_max_bid, suggested_daily_budget,
            shadow_price_lambda, strategic_reasoning,
            constraint_notes, timestamp.
        """
        # Extract current campaign state from the latest snapshot
        latest = state_history[-1]
        current_max_bid = latest.biz_inputs.max_bid
        current_daily_budget = latest.biz_inputs.daily_budget
        current_virtual_day = latest.market_outcome.current_day

        # Step 1 – Contextual Awareness: compute shadow price for this moment
        shadow_lambda = self._compute_shadow_price(analysis_result)

        payload = self._build_user_payload(
            analysis_result, current_max_bid, current_daily_budget,
            shadow_lambda, current_virtual_day,
        )

        # Step 2 – LLM Reasoning: Glass-Box mode selection + bid calculation
        # self._system_prompt carries static policy/knowledge (built once in __init__)
        user_prompt = json.dumps(payload)

        raw_text = self._call_llm([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        raw_decision = self._decode_llm_json(
            raw_text=raw_text,
            analysis_result=analysis_result,
            current_max_bid=current_max_bid,
            current_daily_budget=current_daily_budget,
        )

        # Step 3 – Constraint Mapping: clamp to policy hard limits
        decision = self._apply_constraints(
            raw_decision, current_max_bid, current_daily_budget
        )
        decision = self._apply_performance_guardrails(
            decision, state_history, current_max_bid, current_daily_budget
        )
        decision = self._apply_low_utilization_reset(
            decision, state_history, current_max_bid, current_daily_budget
        )
        decision["shadow_price_lambda"] = round(shadow_lambda, 4)
        decision["timestamp"] = datetime.utcnow().isoformat()

        self._log_decision(decision)
        return decision


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
                    temperature=0.1,
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
                    f"[Strategist] {type(exc).__name__} on attempt {attempt}/{_MAX_ATTEMPTS}. "
                    f"Retrying in {delay:.0f}s…",
                    flush=True,
                )
                time.sleep(delay)

    def _decode_llm_json(
        self,
        raw_text: str,
        analysis_result: Dict[str, Any],
        current_max_bid: float,
        current_daily_budget: float,
    ) -> Dict[str, Any]:
        """
        Parse Strategist LLM output robustly.

        Some responses can occasionally contain malformed JSON despite requesting
        json_object response format. This method attempts safe recovery and falls
        back to a deterministic policy so the simulation never crashes mid-run.
        """
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        extracted = self._extract_json_object(cleaned)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        print(
            "[Strategist] WARNING: Invalid JSON from LLM response. "
            "Using deterministic fallback decision for this cycle.",
            flush=True,
        )
        return self._fallback_raw_decision(
            analysis_result=analysis_result,
            current_max_bid=current_max_bid,
            current_daily_budget=current_daily_budget,
        )

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Extract the first balanced JSON object from arbitrary text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _fallback_raw_decision(
        self,
        analysis_result: Dict[str, Any],
        current_max_bid: float,
        current_daily_budget: float,
    ) -> Dict[str, Any]:
        """Deterministic fallback policy used only when LLM JSON is malformed."""
        signal = str(analysis_result.get("summary_signal", "STABLE")).upper()

        if signal == "TECH_FAILURE":
            mode = "EMERGENCY"
            multiplier = (0.01 / current_max_bid) if current_max_bid > 0 else 1.0
            selected_policies = ["escalation_policy", "execution_constraints"]
        elif signal == "POSITIVE_SURGE":
            mode = "OPPORTUNITY"
            multiplier = 1.10
            selected_policies = ["volatility_response_policy", "risk_appetite_policy"]
        elif signal in {"LEAKAGE_RISK", "NEGATIVE_SHOCK"}:
            mode = "EMERGENCY"
            multiplier = 0.85
            selected_policies = ["cpa_efficiency_policy", "volatility_response_policy"]
        else:
            mode = "NORMAL"
            multiplier = 1.00
            selected_policies = ["risk_appetite_policy", "noise_protection_policy"]

        suggested_max_bid = current_max_bid * multiplier if current_max_bid > 0 else BID_FLOOR
        return {
            "selected_mode": mode,
            "selected_policies": selected_policies,
            "bid_multiplier": float(multiplier),
            "suggested_max_bid": float(suggested_max_bid),
            "suggested_daily_budget": float(current_daily_budget),
            "strategic_reasoning": (
                "Fallback decision executed because LLM output JSON was malformed. "
                f"Signal={signal}, selected_mode={mode}, base_multiplier={multiplier:.4f}."
            ),
        }

    # --------------------------------------------------
    # Shadow Price
    # --------------------------------------------------

    def _compute_shadow_price(self, analysis_result: Dict[str, Any]) -> float:
        """
        λ = exp(t / 24) where t = decimal hour derived from the Analyst's
        timestamp.  Falls back to UTC wall-clock time if no timestamp present.
        Early day → λ close to 1 (prioritise efficiency).
        Late day  → λ up to e ≈ 2.718 (prioritise full budget utilisation).
        """
        ts_str = analysis_result.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                t = ts.hour + ts.minute / 60.0
            except ValueError:
                t = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        else:
            now = datetime.utcnow()
            t = now.hour + now.minute / 60.0
        return math.exp(t / 24.0)

    # --------------------------------------------------
    # Prompt Construction
    # --------------------------------------------------

    def _build_system_prompt(self) -> str:
        return """
You are the Strategist Agent in a Multi-Agent OODA system for a Zurich Audi dealership's digital ad campaign.

Your role:
- Receive the Analyst's structured market signal (summary_signal + reasoning).
- Consult the Policy DB and Shared Knowledge provided in the user message.
- Select ONE operating mode: NORMAL, EMERGENCY, or OPPORTUNITY.
- Compute an exact bid_multiplier.
- Produce transparent "Glass-Box" reasoning that exposes every inference step.

When interpreting the optimization objective, prioritize semantic policy fields
over symbol-only expressions:
- optimization_objective.plain_language_summary
- optimization_objective.objective_breakdown
- optimization_objective.decision_intuition
- utility_intuition.component_explanations
- utility_intuition.decision_guide_for_llms

Use formulas as validation checks, not as the only source of decision logic.

---

RECOVERY MODE OVERRIDE (evaluate this FIRST, before any other mode selection):

If current_max_bid_chf < 4.20 CHF AND summary_signal is NOT "TECH_FAILURE":
  - The system is in a sub-competitive bid state (below the market CPC threshold of ~4.15 CHF).
  - This arises after any bid-pause: emergency exits, daily budget depletion, or crash recovery.
  - At bids below 4.20 CHF, reduced auction volume is EXPECTED — it is NOT a market failure,
    leakage, or negative shock. Activity is proportionally lower simply because fewer auctions
    are won. Below 0.80 CHF, almost no auctions are won; at 2–4 CHF, partial volume is won.
  - OVERRIDE: select NORMAL mode regardless of any other negative signal.
  - Set bid_multiplier = 1.30 (maximum allowed ramp) to accelerate bid recovery to competitive
    levels. Do NOT apply the 0.85× EFFICIENCY_DRIFT correction multiplier — that correction
    fights the recovery ramp and must be suppressed until bids exceed 4.20 CHF.
  - The Taskmaster will apply a recovery floor (4.50 CHF) to jump-start competitiveness.
  - Do NOT select EMERGENCY based on low activity when bids are in the sub-competitive zone.

---

BASELINE HUMAN LOW-UTILISATION BID RESET (12h routine):

Mirror baseline/legacy_human section A behavior:
- Every 12 virtual hours, review rolling 24h spend.
- If rolling_24h_spend < 50% of current_daily_budget_chf:
  - override suggested_max_bid = 5.00 CHF (baseline INITIAL_MAX_BID),
  - regardless of prior multiplier recommendation.
- This reset is a deterministic safety routine and may exceed normal ±30% ramp logic.

MODE DEFINITIONS:
  NORMAL      → Standard shadow pricing. The optimization function F = U + Pacing Score
                governs incremental adjustments (typical range ±5–15%).
  EMERGENCY   → Capital protection mode. Triggered by leakage, tech failure (e.g. website crash), or
                severe CVR shock. Override all performance math; move bid toward
                the policy floor (CHF 0.50). Stop toxic spend first.
  OPPORTUNITY → Confirmed positive CVR surge (e.g., holiday period from event_calendar).
                Aggressive but bounded scaling. Bid multiplier up to 1.30.

MODE SELECTION MAP (primary trigger = summary_signal):
You are to map the Analyst's SIGNAL and SEVERITY to a specific ACTION.

1. NORMAL (Signal: STABLE, EFFICIENCY_DRIFT)
   - Goal: Maintain the OODA path.
   - Math: Max Bid = Base_Bid * exp(t/24).
   - If EFFICIENCY_DRIFT: Apply an additional 0.85x "Correction Multiplier" to the result.

2. EMERGENCY (Signal: TECH_FAILURE, LEAKAGE_RISK, NEGATIVE_SHOCK)
   - If TECH_FAILURE: Mandatory 100% stop. Set Max Bid to 0.01 immediately.
   - If LEAKAGE_RISK or NEGATIVE_SHOCK:
        - Severity 1-4: Do not trigger Emergency. Treat as EFFICIENCY_DRIFT (Normal path + correction).
        - Severity 5-7: Conservative Throttle. Cut Max Bid by 50%.
        - Severity 8-10: Immediate Kill-Switch. Set Max Bid to 0.01.
   - IMPORTANT: Severity must be assessed from actual CPA vs target, NOT from the Analyst's
     confidence_score. The confidence_score measures analytical certainty, not problem severity.
     Use: severity = (actual_CPA - 80) / 8.0, capped at 10.

3. OPPORTUNITY (Signal: POSITIVE_SURGE) — e.g., confirmed holiday demand surge
   - Goal: Capitalize on elevated conversion volume.
   - Math: Max Bid = Base_Bid × 1.25–1.30 (maximum ramp).
   - Holiday bonus: each lead is worth CHF 100 more (per optimization_objective).
   - If current_max_bid < 2.00 CHF: Apply 1.30 multiplier; the Taskmaster recovery
     floor will restore competitive bid levels — prioritize volume capture.
---

GLASS-BOX REASONING REQUIREMENT:
The strategic_reasoning field MUST demonstrate all five reasoning layers:

  1. SIGNAL MAPPING  – State the summary_signal and which mode it triggers, citing
                       the volatility_regime and efficiency_assessment from the Analyst.

  2. OPTIMIZATION MATH – Show your calculation of the optimization function direction:
       - volume_score direction: are leads above or below the implied rate? (ALPHA=500 per lead)
       - cpa_penalty estimate: BETA × max(0, actual_CPA − 80) × leads  (BETA=2, ONE-SIDED)
         Note: zero leads → zero CPA penalty. Below-target CPA is NOT penalised.
       - leakage activation: is CPA > 80 AND CVR < 0.025? (both required)
       - no_conversion penalty: 0.20 × spend  (only when leads == 0 AND spend > 0)
       - shadow price λ interpretation: is it early (λ ≈ 1) or late (λ >> 1)?
       - pacing score: λ × min(spend, target) − 2λ × max(spend − target, 0)
       - Conclude whether F = U + Pacing Score benefits from bid increase, decrease, or hold.

  3. BID MULTIPLIER JUSTIFICATION – State the exact multiplier and explain why:
       (e.g., "Multiplier = 1.10 because pacing is on-track, CVR is stable,
        and λ = 1.67 signals moderate urgency to fill budget without overspending.")

  4. POLICY GROUNDING – Name every policy constraint that shaped or bounded the
       decision (e.g., cpa_efficiency_policy, volatility_response_policy,
       risk_appetite_policy, noise_protection_policy).

  5. UNCERTAINTY ACKNOWLEDGEMENT – Quote the Analyst's uncertainty_notes and state
       whether you acted conservatively or aggressively in response.

---

HARD CONSTRAINTS (enforced programmatically after your output):
  - bid_multiplier:          min = BID_FLOOR / current_max_bid,
                             max = BID_CEILING / current_max_bid,
                             change capped at ±30 % of current bid.

---

Output STRICTLY valid JSON with these fields only:

{
  "selected_mode": "NORMAL | EMERGENCY | OPPORTUNITY",
  "selected_policies": ["<list of policy names invoked, e.g. cpa_efficiency_policy>"],
  "bid_multiplier": <float>,
  "suggested_max_bid": <float — current_max_bid × bid_multiplier, pre-constraint>,
  "suggested_daily_budget": <float — set equal to current_daily_budget_chf>,
  "strategic_reasoning": "<multi-line glass-box explanation covering all 5 layers>"
}

---

POLICY_CONTEXT:
""" + json.dumps({
            "optimization_objective": self.policy.get("optimization_objective"),
            "cpa_efficiency_policy": self.policy.get("cpa_efficiency_policy"),
            "budget_pacing_policy": self.policy.get("budget_pacing_policy"),
            "volatility_response_policy": self.policy.get("volatility_response_policy"),
            "risk_appetite_policy": self.policy.get("risk_appetite_policy"),
            "escalation_policy": self.policy.get("escalation_policy"),
            "noise_protection_policy": self.policy.get("noise_protection_policy"),
            "execution_constraints": self.policy.get("execution_constraints"),
            "transparency_policy": self.policy.get("transparency_policy"),
        }) + """

SHARED_KNOWLEDGE_CONTEXT:
""" + json.dumps({
            "baseline_environment": self.shared_knowledge.get("baseline_environment"),
            "volatility_regimes": self.shared_knowledge.get("volatility_regimes"),
            "market_patterns": self.shared_knowledge.get("market_patterns"),
            "efficiency_relationships": self.shared_knowledge.get("efficiency_relationships"),
            "utility_intuition": self.shared_knowledge.get("utility_intuition"),
            "historical_incidents": self.shared_knowledge.get("historical_incidents"),
            "noise_model_assumptions": self.shared_knowledge.get("noise_model_assumptions"),
            "event_calendar": self.shared_knowledge.get("event_calendar"),
        }) + """
"""

    def _build_user_payload(
        self,
        analysis_result: Dict[str, Any],
        current_max_bid: float,
        current_daily_budget: float,
        shadow_lambda: float,
        current_virtual_day: int,
    ) -> Dict[str, Any]:
        """Package only dynamic per-call data; static policy/knowledge is in the system prompt."""
        return {
            "analyst_signal": analysis_result,
            "current_campaign_state": {
                "current_max_bid_chf": current_max_bid,
                "current_daily_budget_chf": current_daily_budget,
                "shadow_price_lambda": round(shadow_lambda, 4),
                "implied_hourly_budget_chf": round(current_daily_budget / 24.0, 2),
                "bid_floor_chf": BID_FLOOR,
                "bid_ceiling_chf": BID_CEILING,
                "max_bid_change_pct_per_hour": MAX_BID_CHANGE_PCT,
                "current_virtual_day": current_virtual_day,
            },
        }

    # --------------------------------------------------
    # Constraint Enforcement
    # --------------------------------------------------

    def _apply_constraints(
        self,
        raw: Dict[str, Any],
        current_max_bid: float,
        current_daily_budget: float,
    ) -> Dict[str, Any]:
        """
        Programmatically enforce all hard policy limits on the LLM's suggested
        values.  Any clamping is recorded in constraint_notes for full auditability.
        """
        notes: List[str] = []

        # ── Bid Multiplier ──────────────────────────────────────────────────
        raw_multiplier = float(raw.get("bid_multiplier", 1.0))

        # Per-hour change cap (±30 %)
        max_up = 1.0 + MAX_BID_CHANGE_PCT    # 1.30
        max_dn = 1.0 - MAX_BID_CHANGE_PCT    # 0.70

        multiplier = raw_multiplier
        if multiplier > max_up:
            multiplier = max_up
            notes.append(
                f"bid_multiplier clamped {raw_multiplier:.3f} → {max_up:.3f} "
                f"(risk_appetite_policy: max +{int(MAX_BID_CHANGE_PCT*100)}%/hr)"
            )
        elif multiplier < max_dn:
            multiplier = max_dn
            notes.append(
                f"bid_multiplier clamped {raw_multiplier:.3f} → {max_dn:.3f} "
                f"(risk_appetite_policy: max -{int(MAX_BID_CHANGE_PCT*100)}%/hr)"
            )

        # Absolute bid floor / ceiling
        suggested_bid = current_max_bid * multiplier
        if suggested_bid < BID_FLOOR:
            suggested_bid = BID_FLOOR
            multiplier = BID_FLOOR / current_max_bid
            notes.append(
                f"suggested_max_bid floored at CHF {BID_FLOOR} "
                f"(execution_constraints: no_negative_bids)"
            )
        elif suggested_bid > BID_CEILING:
            suggested_bid = BID_CEILING
            multiplier = BID_CEILING / current_max_bid
            notes.append(
                f"suggested_max_bid capped at CHF {BID_CEILING} "
                f"(execution_constraints: respect_market_ceiling)"
            )

        # ── Daily Budget ────────────────────────────────────────────────────
        # Budget authority is owned by Zupervisor; Strategist emits only
        # passthrough telemetry value for compatibility.
        raw_budget = float(raw.get("suggested_daily_budget", current_daily_budget))
        budget = float(current_daily_budget)
        if abs(raw_budget - current_daily_budget) > 0.01:
            notes.append(
                "suggested_daily_budget ignored (budget authority delegated to Zupervisor)."
            )

        return {
            "selected_mode": raw.get("selected_mode", "NORMAL"),
            "selected_policies": raw.get("selected_policies", []),
            "bid_multiplier": round(multiplier, 4),
            "suggested_max_bid": round(suggested_bid, 2),
            "suggested_daily_budget": round(budget, 2),
            "strategic_reasoning": raw.get("strategic_reasoning", ""),
            "constraint_notes": notes if notes else ["No constraints violated."],
        }


    def _apply_low_utilization_reset(
        self,
        decision: Dict[str, Any],
        state_history: List[SimulationState],
        current_max_bid: float,
        current_daily_budget: float,
    ) -> Dict[str, Any]:
        """
        Mirror baseline human 12h routine check:
        if rolling 24h spend is below 50% of current budget, reset bid to 5.00 CHF.
        """
        if not state_history:
            return decision

        current_hour_abs = state_history[-1].market_outcome.current_hour
        if current_hour_abs <= 0 or current_hour_abs % LOW_UTIL_CHECK_INTERVAL_HOURS != 0:
            return decision

        history_24h = state_history[-LOOKBACK_STEPS_24H:]
        if not history_24h:
            return decision

        rolling_spend = sum(s.market_outcome.spend for s in history_24h)
        if rolling_spend >= (current_daily_budget * 0.5):
            return decision

        decision["suggested_max_bid"] = round(INITIAL_MAX_BID, 2)
        if current_max_bid > 0:
            decision["bid_multiplier"] = round(INITIAL_MAX_BID / current_max_bid, 4)
        notes = decision.get("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
        notes.append(
            "baseline_low_utilization_reset: rolling_24h_spend < 50% budget, bid reset to CHF 5.00."
        )
        decision["constraint_notes"] = notes

        reasoning = decision.get("strategic_reasoning", "")
        decision["strategic_reasoning"] = (
            reasoning
            + ("\n" if reasoning else "")
            + "Baseline low-utilization reset applied: rolling 24h spend below 50% budget, bid reset to CHF 5.00."
        )
        return decision

    def _apply_performance_guardrails(
        self,
        decision: Dict[str, Any],
        state_history: List[SimulationState],
        current_max_bid: float,
        current_daily_budget: float,
    ) -> Dict[str, Any]:
        """
        Deterministic anti-oscillation layer to keep bid moves economically sane.

        The LLM can still choose mode and direction, but this guardrail ensures we
        do not repeatedly overbid when 24h efficiency is clearly deteriorating.
        """
        if not state_history or current_max_bid <= 0:
            return decision

        window = state_history[-LOOKBACK_STEPS_24H:]
        if not window:
            return decision

        spend_24h = sum(s.market_outcome.spend for s in window)
        leads_24h = sum(s.market_outcome.leads for s in window)
        clicks_24h = sum(s.market_outcome.clicks for s in window)
        if clicks_24h < MIN_GUARDRAIL_CLICKS:
            return decision

        cpa_24h = (spend_24h / leads_24h) if leads_24h > 0 else float("inf")
        spend_ratio = (spend_24h / current_daily_budget) if current_daily_budget > 0 else 0.0

        mode = str(decision.get("selected_mode", "NORMAL")).upper()
        multiplier = float(decision.get("bid_multiplier", 1.0))
        notes = decision.get("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]

        if mode == "NORMAL":
            bounded = max(NORMAL_MODE_MAX_DN, min(multiplier, NORMAL_MODE_MAX_UP))
            if bounded != multiplier:
                notes.append(
                    f"normal_mode_smoothing: multiplier clamped {multiplier:.3f} → {bounded:.3f}."
                )
                multiplier = bounded
        elif mode == "OPPORTUNITY":
            bounded = min(multiplier, OPPORTUNITY_MODE_MAX_UP)
            if bounded != multiplier:
                notes.append(
                    f"opportunity_mode_ceiling: multiplier clamped {multiplier:.3f} → {bounded:.3f}."
                )
                multiplier = bounded

        # Efficiency-first correction: if 24h CPA is poor, never allow bid increases.
        if cpa_24h >= CPA_HARD_CUT or (leads_24h == 0 and spend_24h > 1.5 * CPA_SOFT_CUT):
            bounded = min(multiplier, 0.80)
            if bounded != multiplier:
                notes.append(
                    f"efficiency_hard_cut: 24h CPA={cpa_24h:.2f}, forcing multiplier ≤ {bounded:.2f}."
                )
                multiplier = bounded
        elif cpa_24h >= CPA_SOFT_CUT:
            bounded = min(multiplier, 0.92)
            if bounded != multiplier:
                notes.append(
                    f"efficiency_soft_cut: 24h CPA={cpa_24h:.2f}, forcing multiplier ≤ {bounded:.2f}."
                )
                multiplier = bounded

        # Pacing protection: if we already overspent the rolling budget, suppress up-ramps.
        if spend_ratio > 1.05:
            bounded = min(multiplier, 0.90)
            if bounded != multiplier:
                notes.append(
                    f"pacing_cut: rolling spend ratio={spend_ratio:.2f}, forcing multiplier ≤ {bounded:.2f}."
                )
                multiplier = bounded

        suggested_bid = max(BID_FLOOR, min(current_max_bid * multiplier, BID_CEILING))
        if abs(suggested_bid - current_max_bid) < 0.01:
            suggested_bid = current_max_bid
            multiplier = 1.0

        decision["bid_multiplier"] = round(multiplier, 4)
        decision["suggested_max_bid"] = round(suggested_bid, 2)
        decision["constraint_notes"] = notes
        return decision

    # --------------------------------------------------
    # Policy Reload
    # --------------------------------------------------

    def reload_policy(self) -> None:
        """Re-read policy_db.json from disk to pick up Zupervisor edits."""
        self.policy = self._load_json(self.policy_path)
        # Rebuild system prompt so updated alpha/beta from Zupervisor take effect.
        self._system_prompt = self._build_system_prompt()

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _log_decision(self, decision: Dict[str, Any]) -> None:
        """Append the decision as a JSONL record so the Taskmaster can read it."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(decision) + "\n")

## IGNORE THIS CODE BELOW ##
# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load the latest Analyst signal from the JSONL log
# _analyst_log_path = os.path.join(_SCRIPT_DIR, "logs", "analyst_log.jsonl")
# with open(_analyst_log_path, "r") as _f:
#     _latest_analysis = json.loads(_f.readlines()[-1])

# test_strategist = Strategist(
#     shared_knowledge_path=os.path.join(_SCRIPT_DIR, "knowledge", "shared_knowledge.json"),
#     policy_path=os.path.join(_SCRIPT_DIR, "knowledge", "policy_db.json"),
#     log_path=os.path.join(_SCRIPT_DIR, "logs", "strategist_log.jsonl"),
# )

# test_strategist.decide(
#     analysis_result=_latest_analysis,
#     current_max_bid=2.0,
#     current_daily_budget=1000.0,
# )
## IGNORE THIS CODE ABOVE ##