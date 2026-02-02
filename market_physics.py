import numpy as np


class MarketPhysics:
    """
    Translates the configuration stored in `StateManager` into realized outcomes.

    This class does **not** own any state about bids or volatility itself:
    it always **reads** from the `StateManager` and **writes** outcomes back
    via `StateManager.record_outcome(...)`.
    """

    def __init__(
        self,
        cpc_base: float = 1.25,
        cpc_max: float = 8.00,
        cvr_base: float = 0.025,
        k: float = 1.2,
        base_clicks: int = 5,
    ):
        # Market constants
        self.CPC_base = float(cpc_base)
        self.CPC_max = float(cpc_max)
        self.CVR_base = float(cvr_base)
        self.k = float(k)
        self.base_clicks = int(base_clicks)

    def run_step(self, state_manager) -> dict:
        """
        Read inputs from the state manager, generate stochastic outcomes,
        and write them back to the state manager with a timestamp.
        """
        inputs = state_manager.get_inputs()
        b_max = float(inputs["max_bid"])
        v_multiplier = float(inputs["volatility"])

        # Guard against degenerate configs
        if b_max <= 0.0:
            outcome = {
                "realized_cpc": 0.0,
                "spend": 0.0,
                "leads": 0,
                "clicks": 0,
                "cpa": 0.0,
            }
            state_manager.record_outcome(outcome)
            return outcome

        # 1) Actual CPC (asymptotic saturation in bid)
        cpc_act = self.CPC_base + (self.CPC_max - self.CPC_base) * (
            1 - np.exp(-self.k * (b_max / self.CPC_max))
        )

        # 2) Traffic volume (scaled by bid) + log-normal-ish noise
        clicks = int(
            self.base_clicks
            * (b_max / self.CPC_base)
            * np.random.normal(1.0, 0.05)
        )
        clicks = max(0, clicks)

        # 3) Spend and leads
        spend = clicks * cpc_act
        cvr_effective = self.CVR_base * v_multiplier
        leads = int(np.random.poisson(clicks * max(cvr_effective, 0.0)))

        outcome = {
            "realized_cpc": round(float(cpc_act), 4),
            "spend": round(float(spend), 4),
            "leads": int(leads),
            "clicks": int(clicks),
        }
        outcome["cpa"] = (
            round(float(spend) / float(leads), 4) if leads > 0 else round(float(spend), 4)
        )

        # Write through to StateManager (single source of truth)
        state_manager.record_outcome(outcome)
        return outcome

