"""Policy selection & Shadow Pricing (exp(t/24)) agent for OODA MAS."""

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

BID_FLOOR = 0.50       # CHF – absolute minimum bid (mirrors rule_engine)
BID_CEILING = 20.00    # CHF – absolute maximum bid (mirrors rule_engine)
MAX_BID_CHANGE_PCT = 0.30   # ±30 % per hour (risk_appetite_policy)
MAX_BUDGET_CHANGE_PCT = 0.20  # ±20 % per cycle (escalation_policy)


class Strategist:
    """
    Strategist agent: Policy Selection and Shadow Pricing.

    Consumes the Analyst's structured signal, consults the policy_db and
    shared_knowledge_db, and uses an LLM (GPT) to select one of three
    operating modes:

      NORMAL      – Standard shadow pricing; F = U - λ·P governs bid adjustments.
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

        self.shared_knowledge = self._load_json(shared_knowledge_path)
        self.policy = self._load_json(policy_path)
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

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
        current_max_bid: float = 2.0,
        current_daily_budget: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Produce a strategic bidding decision from the Analyst's signal.

        Parameters
        ----------
        analysis_result : dict
            Full output from Analyst.analyze() – contains analysis_result,
            reasoning, confidence_score, summary_signal, and timestamp.
        current_max_bid : float
            The active max-bid (CPC cap) currently configured in the sandbox.
        current_daily_budget : float
            The active daily budget currently configured in the sandbox.

        Returns
        -------
        dict
            selected_mode, selected_policies, bid_multiplier,
            suggested_max_bid, suggested_daily_budget,
            shadow_price_lambda, strategic_reasoning,
            constraint_notes, timestamp.
        """
        # Step 1 – Contextual Awareness: compute shadow price for this moment
        shadow_lambda = self._compute_shadow_price(analysis_result)

        payload = self._build_user_payload(
            analysis_result, current_max_bid, current_daily_budget, shadow_lambda
        )

        # Step 2 – LLM Reasoning: Glass-Box mode selection + bid calculation
        system_prompt = self._build_system_prompt()
        user_prompt = json.dumps(payload, indent=2)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_decision = json.loads(response.choices[0].message.content)

        # Step 3 – Constraint Mapping: clamp to policy hard limits
        decision = self._apply_constraints(
            raw_decision, current_max_bid, current_daily_budget
        )
        decision["shadow_price_lambda"] = round(shadow_lambda, 4)
        decision["timestamp"] = datetime.utcnow().isoformat()

        self._log_decision(decision)
        return decision

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
- Compute an exact bid_multiplier and a suggested_daily_budget.
- Produce transparent "Glass-Box" reasoning that exposes every inference step.

---

MODE DEFINITIONS:
  NORMAL      → Standard shadow pricing. The optimization function F = U - λ·P
                governs incremental adjustments (typical range ±5–15%).
  EMERGENCY   → Capital protection mode. Triggered by leakage, tech failure, or
                severe CVR shock. Override all performance math; move bid toward
                the policy floor (CHF 0.50). Stop toxic spend first.
  OPPORTUNITY → Confirmed positive CVR surge. Aggressive but bounded scaling.
                Bid multiplier may reach up to 1.30; never exceed policy ceiling.

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
        - Severity 5-7: Conservative Throttle. Cut Max Bid by 50% and reduce Daily Budget.
        - Severity 8-10: Immediate Kill-Switch. Set Max Bid to 0.01.

3. OPPORTUNITY (Signal: POSITIVE_SURGE)
   - Goal: Capitalize on volume.
---

GLASS-BOX REASONING REQUIREMENT:
The strategic_reasoning field MUST demonstrate all five reasoning layers:

  1. SIGNAL MAPPING  – State the summary_signal and which mode it triggers, citing
                       the volatility_regime and efficiency_assessment from the Analyst.

  2. OPTIMIZATION MATH – Show your calculation of the optimization function direction:
       - volume_score direction: are leads above or below the implied rate?
       - cpa_penalty estimate: 5 × |observed_CPA - 80|  (use efficiency_assessment clues)
       - leakage activation: is CPA > 80 AND CVR < 0.025?
       - shadow price λ interpretation: is it early (λ ≈ 1) or late (λ >> 1)?
       - Conclude whether F benefits from bid increase, decrease, or hold.

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
  - suggested_daily_budget:  within ±20 % of current_daily_budget.

---

Output STRICTLY valid JSON with these fields only:

{
  "selected_mode": "NORMAL | EMERGENCY | OPPORTUNITY",
  "selected_policies": ["<list of policy names invoked, e.g. cpa_efficiency_policy>"],
  "bid_multiplier": <float>,
  "suggested_max_bid": <float — current_max_bid × bid_multiplier, pre-constraint>,
  "suggested_daily_budget": <float>,
  "strategic_reasoning": "<multi-line glass-box explanation covering all 5 layers>"
}
"""

    def _build_user_payload(
        self,
        analysis_result: Dict[str, Any],
        current_max_bid: float,
        current_daily_budget: float,
        shadow_lambda: float,
    ) -> Dict[str, Any]:
        """Package all context the LLM needs for fully-grounded reasoning."""
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
            },
            "policy_context": {
                "optimization_objective": self.policy.get("optimization_objective"),
                "cpa_efficiency_policy": self.policy.get("cpa_efficiency_policy"),
                "budget_pacing_policy": self.policy.get("budget_pacing_policy"),
                "volatility_response_policy": self.policy.get("volatility_response_policy"),
                "risk_appetite_policy": self.policy.get("risk_appetite_policy"),
                "escalation_policy": self.policy.get("escalation_policy"),
                "noise_protection_policy": self.policy.get("noise_protection_policy"),
                "execution_constraints": self.policy.get("execution_constraints"),
                "transparency_policy": self.policy.get("transparency_policy"),
            },
            "shared_knowledge_context": {
                "baseline_environment": self.shared_knowledge.get("baseline_environment"),
                "volatility_regimes": self.shared_knowledge.get("volatility_regimes"),
                "market_patterns": self.shared_knowledge.get("market_patterns"),
                "efficiency_relationships": self.shared_knowledge.get("efficiency_relationships"),
                "utility_intuition": self.shared_knowledge.get("utility_intuition"),
                "historical_incidents": self.shared_knowledge.get("historical_incidents"),
                "noise_model_assumptions": self.shared_knowledge.get("noise_model_assumptions"),
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
        raw_budget = float(raw.get("suggested_daily_budget", current_daily_budget))
        budget_min = current_daily_budget * (1.0 - MAX_BUDGET_CHANGE_PCT)
        budget_max = current_daily_budget * (1.0 + MAX_BUDGET_CHANGE_PCT)
        budget = max(budget_min, min(raw_budget, budget_max))

        if abs(budget - raw_budget) > 0.01:
            notes.append(
                f"suggested_daily_budget clamped CHF {raw_budget:.2f} → CHF {budget:.2f} "
                f"(escalation_policy: max ±{int(MAX_BUDGET_CHANGE_PCT*100)}% deviation)"
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