"""Observation / Anomaly Detection agent for OODA MAS."""

from typing import List, Optional

import json
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from openai import OpenAI


class Analyst:
    """
    Analyst agent: Observation and Anomaly Detection.

    This agent:
    - Interprets recent performance context
    - Uses shared knowledge + policy grounding
    - Produces structured analytical output
    - Does NOT make decisions
    """

    def __init__(
        self,
        shared_knowledge_path: str,
        policy_path: str,
        log_path: str = "logs/analyst_log.jsonl",
        model: str = "gpt-4o-mini"
    ):
        load_dotenv()
        print(f"Key loaded: {os.getenv('OPENAI_API_KEY')[:5]}...")
        self.client = OpenAI()
        self.model = model

        self.shared_knowledge = self._load_json(shared_knowledge_path)
        self.policy = self._load_json(policy_path)
        self.log_path = log_path

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    # --------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------

    def analyze(
        self,
        historical_df: pd.DataFrame,
        tech_ping: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        # Recent context window (last 1 hour = 4 steps)
        window_df = historical_df.tail(4)
        loopback_df = historical_df.tail(16)  # 4 hours context
        rolling_24h_lookback_df = historical_df.tail(96)  # 24 hours context

        payload = {
            "recent_window_data": window_df.to_dict(orient="records"),
            "loopback_context": loopback_df.to_dict(orient="records"),
            "rolling_24h_lookback_context": rolling_24h_lookback_df.to_dict(orient="records"),
            "technology_ping": tech_ping,
            "shared_knowledge": self._extract_relevant_knowledge(),
            "policy_context": self._extract_relevant_policy()
        }

        system_prompt = self._build_system_prompt()
        user_prompt = json.dumps(payload, indent=2)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        analysis_output = json.loads(response.choices[0].message.content)

        # Attach timestamp
        analysis_output["timestamp"] = datetime.utcnow().isoformat()

        self._log_analysis(analysis_output)

        return analysis_output

    # --------------------------------------------------
    # Prompt Construction
    # --------------------------------------------------

    def _build_system_prompt(self) -> str:
        return """
You are the Analyst Agent in a Multi-Agent OODA system.

Your role:
- Observe performance metrics.
- Contextualize them using shared knowledge and policy.
- Detect patterns, anomalies, and regimes.
- Assess efficiency, leakage risk, pacing status, and technology impact.
- DO NOT suggest actions.
- DO NOT make decisions.
- Only analyze and interpret.

---

HOW TO USE SHARED KNOWLEDGE:
- baseline_environment: Use target_cpa, baseline_cvr, baseline_cpc, market_ceiling_cpc to assess deviations.
- market_patterns: Match observed behavior to patterns (holiday_surge, landing_page_failure, competitive_bid_spike, normal_noise).
- historical_incidents: Reference lessons learned (cvr_crash_overreaction, late_day_underspend, holiday_underbidding) to avoid misclassification.
- volatility_regimes: Classify into stable, negative_shock, or positive_surge using cvr_multiplier_range.
- efficiency_relationships: Apply leakage_condition (CPA > target AND CVR < baseline = toxic spend).
- noise_model_assumptions: Require confirmation before treating single-hour moves as structural; avoid overreaction.
- utility_intuition: Interpret metrics in terms of lead_value, cpa_penalty_weight, leakage_penalty_weight.

HOW TO USE POLICY CONTEXT:
- optimization_objective: F = U - λ·P; understand volume_score, cpa_penalty, leakage_penalty, budget_penalty.
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
    "uncertainty_notes": "..."
  },
  "confidence_score": 0.0-1.0,
  "summary_signal": "ONE_OF: STABLE | NEGATIVE_SHOCK | POSITIVE_SURGE | LEAKAGE_RISK | TECH_FAILURE | EFFICIENCY_DRIFT"
}

Guidelines:
- Use shared knowledge to classify regime and match market patterns.
- Reference policy thresholds when assessing efficiency, pacing, and escalation risk.
- Distinguish noise from structural change using noise_model_assumptions and historical_incidents.
- Be economically rational and remain neutral and analytical.
"""

    def _extract_relevant_knowledge(self) -> Dict[str, Any]:
        """
        Extract shared knowledge components relevant for analysis.
        Uses baseline, patterns, incidents, regimes, and relationships.
        """
        return {
            "baseline_environment": self.shared_knowledge.get("baseline_environment"),
            "market_patterns": self.shared_knowledge.get("market_patterns"),
            "historical_incidents": self.shared_knowledge.get("historical_incidents"),
            "efficiency_relationships": self.shared_knowledge.get("efficiency_relationships"),
            "utility_intuition": self.shared_knowledge.get("utility_intuition"),
            "volatility_regimes": self.shared_knowledge.get("volatility_regimes"),
            "noise_model_assumptions": self.shared_knowledge.get("noise_model_assumptions"),
            "stakeholder_expectations": self.shared_knowledge.get("stakeholder_expectations"),
        }

    def _extract_relevant_policy(self) -> Dict[str, Any]:
        """
        Extract policy components relevant for analysis.
        Uses optimization objective, efficiency, pacing, volatility, risk, escalation, and noise rules.
        """
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


## IGNORE THIS CODE BELOW ##
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

test_analyst = Analyst(
    shared_knowledge_path=os.path.join(_SCRIPT_DIR, "knowledge", "shared_knowledge.json"),
    policy_path=os.path.join(_SCRIPT_DIR, "knowledge", "policy_db.json"),
    log_path=os.path.join(_SCRIPT_DIR, "logs", "analyst_log.jsonl"),
)

test_analyst.analyze(
    historical_df=pd.read_csv("data/ib_results.csv"),
    tech_ping=None,
)
## IGNORE THIS CODE ABOVE ##    