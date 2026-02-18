"""Observation / Anomaly Detection agent for OODA MAS."""

from typing import List, Optional

from core.state_manager import SimulationState


class Analyst:
    """
    Analyst agent: Observation and Anomaly Detection.

    Observes state history, detects anomalies (e.g., CPA spikes,
    budget depletion, volatility events), and surfaces signals
    for the Strategist and Human Supervisor.
    """

    def __init__(self):
        pass

    def observe(self, state_history: List[SimulationState]) -> dict:
        """
        Analyze state history and return observation summary.

        Returns a dict with metrics and anomaly flags.
        """
        if not state_history:
            return {"anomalies": [], "metrics": {}}

        latest = state_history[-1]
        # Placeholder: extract key metrics
        metrics = {
            "current_cpa": latest.market_outcome.cpa,
            "current_spend": latest.market_outcome.spend,
            "current_leads": latest.market_outcome.leads,
            "budget_status": latest.derived_variables.budget_status,
            "volatility": latest.external_events_inputs.volatility,
        }
        anomalies = []
        # TODO: Implement anomaly detection logic
        return {"anomalies": anomalies, "metrics": metrics}
