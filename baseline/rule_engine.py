"""Proportional rules (Bid +/- 10%) for Industry Baseline automation."""

from typing import List

from core.sandbox_env import SandboxEnv
from core.state_manager import SimulationState

# Proportional rule parameters
LOOKBACK_STEPS = 96          # 24 hours
MIN_CLICKS_REQUIRED = 10     # avoid reacting to noise
BID_FLOOR = 0.50             # never go below this
BID_CEILING = 20.00          # protect against runaway bids


def apply_proportional_rule(
    env: SandboxEnv,
    state_history: List[SimulationState],
    target_cpa: float,
) -> None:
    """
    Realistic Hybrid Rule Engine:
    1. Lookback Window: Uses a rolling 24-hour average to smooth noise.
    2. Statistical Significance: Only acts if there is enough data (min_clicks).
    3. Efficiency Rule: Lowers bid if CPA is too high.
    4. Volume Rule: Aggressively increases bid if CPA is low AND spend is under-pacing.
    5. Inventory Guard: Prevents bids from dropping below a floor or exceeding a ceiling.
    """

    # 1. Extract historical data from the rolling window
    window = state_history[-LOOKBACK_STEPS:]
    if not window:
        return

    # 2. Calculate aggregate metrics over the window
    total_spend = sum(s.market_outcome.spend for s in window)
    total_leads = sum(s.market_outcome.leads for s in window)
    total_clicks = sum(s.market_outcome.clicks for s in window)

    # Current config from the latest state snapshot
    latest = state_history[-1]
    current_max_bid = latest.biz_inputs.max_bid
    daily_budget = latest.biz_inputs.daily_budget

    # 3. Statistical guardrail – do nothing if we don't have enough data
    if total_clicks < MIN_CLICKS_REQUIRED:
        return

    # --- A. ROLLING PACING MULTIPLIER ---
    # We treat the 'daily_budget' as the target spend for any 24h window.
    pacing_multiplier = 1.0
    if total_spend < (daily_budget * 0.85):
        # Under-spending over the last 24h: increase bid to find volume
        pacing_multiplier = 1.10
    elif total_spend > (daily_budget * 1.10):
        # Over-spending over the last 24h: decrease bid to conserve
        pacing_multiplier = 0.90

    # --- B. ROLLING EFFICIENCY MULTIPLIER ---
    efficiency_multiplier = 1.0
    if total_leads > 0:
        avg_cpa = total_spend / total_leads
        if avg_cpa >= target_cpa * 1.3:
            efficiency_multiplier = 0.75  # Aggressive cut
        elif avg_cpa >= target_cpa * 1.1:
            efficiency_multiplier = 0.85  # Moderate cut
        elif avg_cpa > target_cpa:
            efficiency_multiplier = 0.95  # Minor trim
        elif avg_cpa < target_cpa * 0.8:
            efficiency_multiplier = 1.10  # Scale up
    else:
        # Death Spiral Protection: Only cut if we spent > 1.5x target with 0 leads
        if total_spend > (target_cpa * 1.5):
            efficiency_multiplier = 0.70

    new_max_bid = current_max_bid * pacing_multiplier * efficiency_multiplier

    # Enforce physical boundaries
    new_max_bid = max(BID_FLOOR, min(new_max_bid, BID_CEILING))

    # Execute only if there's a meaningful change
    if round(new_max_bid, 2) != round(current_max_bid, 2):
        env.configure(max_bid=new_max_bid)
