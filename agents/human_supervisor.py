"""The 3-7 day 'Strategic Guidance' agent for OODA MAS."""

from typing import Optional


class HumanSupervisor:
    """
    Human Supervisor agent: Strategic Guidance on 3-7 day cadence.

    Provides high-level strategic signals (e.g., "scale up for holiday",
    "conserve during crash") that the Strategist incorporates into
    policy selection. Operates on a slower cycle than the Analyst/Strategist.
    """

    def __init__(self):
        pass

    def get_guidance(
        self,
        current_day: int,
        observation_summary: dict,
    ) -> Optional[dict]:
        """
        Return strategic guidance based on calendar and performance.

        Called every 3-7 days. Returns dict with signals like:
        - "scale_up" | "scale_down" | "hold"
        - "holiday_approaching" | "crash_recovery" | None
        """
        # TODO: Implement 3-7 day strategic guidance logic
        return None
