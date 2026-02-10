import numpy as np

# Market physics constants for a premium car dealership.
# CPC_base: base CPC for the car dealership.
CPC_BASE = 2.00
# CPC_max: max CPC for the car dealership.
CPC_MAX = 12.00
# CVR_base: base CVR for the car dealership.
CVR_BASE = 0.025
# k: decay rate for the CPC.
K = 1.2
# base_clicks: base number of clicks for the car dealership (every 15 minutes).
BASE_CLICKS = 5.0


class MarketPhysics:
    """
    Translates the configuration stored in `StateManager` into realized outcomes.

    This class does **not** own any state about bids or volatility itself:
    it always **reads** from the `StateManager` and **writes** outcomes back
    via `StateManager.record_outcome(...)`.
    """

    def __init__(
        self,
        cpc_base: float = CPC_BASE,
        cpc_max: float = CPC_MAX,
        cvr_base: float = CVR_BASE,
        k: float = K,
        # simulate 5 click per step (every 15 minutes)
        base_clicks: float = BASE_CLICKS,
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
        biz_inputs, external_events_inputs = state_manager.get_inputs()
        b_max = float(biz_inputs.max_bid)
        v_multiplier = float(external_events_inputs.volatility)
        daily_budget = float(biz_inputs.daily_budget)
        current_day_spend = float(
            state_manager.state.derived_variables.current_day_spend
        )

        # Check budget status (derived variable)
        if state_manager.state.derived_variables.budget_status == "budget_depleted":
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

        # Cap spend at remaining daily budget so we never overshoot
        remaining_daily = daily_budget - current_day_spend
        if spend > remaining_daily:
            spend = max(remaining_daily, 0.0)
            clicks = spend / cpc_act if cpc_act > 0 else 0.0
            clicks = int(clicks)

        cvr_effective = self.CVR_base * v_multiplier
        leads = np.random.binomial(n=int(clicks), p=min(max(cvr_effective, 0.0), 1.0))

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

