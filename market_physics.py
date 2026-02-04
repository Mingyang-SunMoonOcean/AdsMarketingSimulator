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
        cpc_base: float = 2.00,
        cpc_max: float = 20.00,
        cvr_base: float = 0.025,
        k: float = 1.2,
        # simulate 1 click per step (every 15 minutes)
        base_clicks: float = 1.0,
    ):
        # Market constants
        self.CPC_base = float(cpc_base)
        self.CPC_max = float(cpc_max)
        self.CVR_base = float(cvr_base)
        self.k = float(k)
        self.base_clicks = float(base_clicks)

    def run_step(self, state_manager) -> dict:
        """
        Read inputs from the state manager, generate stochastic outcomes,
        and write them back to the state manager with a timestamp.
        """
        inputs = state_manager.get_inputs()
        b_max = float(inputs["max_bid"])
        v_multiplier = float(inputs["volatility"])
        history = state_manager.state.history

        # Check last 24 hours of spend
        if state_manager.state.budget_status == "budget_depleted":
            outcome = {
                "realized_cpc": 0.0,
                "spend": 0.0,
                "leads": 0,
                "clicks": 0,
                "cpa": 0.0,
            }
            state_manager.record_outcome(outcome)
            return outcome

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
        cpc_act = np.random.uniform(self.CPC_base + (self.CPC_max - self.CPC_base) * (
            1 - np.exp(-self.k * (b_max / self.CPC_max))
        ))
        cpc_act = min(cpc_act, b_max)

        # 2) Traffic volume (scaled by bid) + log-normal-ish noise
        clicks = float(np.random.poisson(self.base_clicks
            * (b_max / self.CPC_base)
            * np.random.normal(1.0, 0.10))
        )
        clicks = max(0, clicks)

        # 3) Spend and leads
        spend = float(clicks) * float(cpc_act)
        cvr_effective = self.CVR_base * v_multiplier
        leads = np.random.binomial(n=int(clicks), p=max(cvr_effective, 0.0))

        outcome = {
            "realized_cpc": round(float(cpc_act), 4),
            "spend": round(float(spend), 4),
            "leads": float(leads),
            "clicks": float(clicks),
        }
        outcome["cpa"] = (
            round(float(spend) / float(leads), 4) if leads > 0 else round(float(spend), 4)
        )

        # Write through to StateManager (single source of truth)
        state_manager.record_outcome(outcome)
        return outcome

