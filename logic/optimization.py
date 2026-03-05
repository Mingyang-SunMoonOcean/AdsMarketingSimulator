"""Total Optimization Function F = U + Pacing Score for OODA MAS."""

import math

import pandas as pd


# ---------------------------------------------------------------------------
# Economic Parameters
# ---------------------------------------------------------------------------
ALPHA = 500.0            # CHF gross margin value per lead
BONUS_VALUE_PER_LEAD = 100  # Extra strategic value per lead during holiday
BETA = 2.0               # CPA-excess penalty multiplier (risk-aversion weight)
GAMMA = 1.5              # Leakage multiplier (toxic-spend penalty weight)
OVERSPEND_FACTOR = 2.0   # Overspend penalised at 2× the pacing reward rate
KAPPA = 0.20             # No-conversion overhead rate: 20% of spend with 0 leads
TARGET_CPA = 80.0        # CHF — target cost-per-acquisition
BASE_CVR = 0.025         # Baseline conversion rate (healthy steady-state)


# ---------------------------------------------------------------------------
# Utility: U = volume_score − cpa_penalty − leakage_penalty
# ---------------------------------------------------------------------------

def calculate_utility(
    window_df: pd.DataFrame,
    volume_bonus: float = 0.0,
) -> float:
    """
    Economic utility for a window of simulation steps.

    Components
    ----------
    volume_score    : ALPHA * leads  (+ holiday bonus)
                      Gross margin from acquired leads.

    cpa_penalty     : BETA * max(0, actual_CPA − TARGET_CPA) * leads
                      Penalises only ABOVE-target CPA, scaled by lead volume.
                      Rationale — zero leads → zero CPA activity → zero penalty.
                      Below-target CPA is efficient and is not penalised.
                      Multiplying by leads converts excess-cost-per-lead into
                      total excess cost, amplified by BETA as a risk-aversion weight.

    leakage_penalty : GAMMA * spend * performance_gap
                      Penalises toxic spend — budget consumed when CVR is below
                      baseline (CPA > target AND CVR < baseline).
    """
    leads = window_df["leads"].sum()
    spend = window_df["spend"].sum()
    clicks = window_df["clicks"].sum()

    actual_cpa = spend / leads if leads > 0 else 0.0
    actual_cvr = leads / clicks if clicks > 0 else 0.0

    volume_score = (ALPHA + volume_bonus * BONUS_VALUE_PER_LEAD) * leads

    # Scaled, one-sided: only penalises CPA above target, proportional to volume
    cpa_penalty = BETA * max(0.0, actual_cpa - TARGET_CPA) * leads

    # Toxic spend: CPA above target AND CVR below baseline
    leakage_penalty = 0.0
    if leads > 0 and actual_cpa > TARGET_CPA:
        performance_ratio = actual_cvr / BASE_CVR
        performance_gap = max(0.0, 1.0 - performance_ratio)
        leakage_penalty = GAMMA * spend * performance_gap

    # No-conversion overhead: spend occurred but zero leads were generated.
    # Applies a smooth, proportional friction (KAPPA × spend) rather than
    # the original flat per-window hammer.  Scales naturally — low-spend
    # quiet windows barely feel it; large-spend-with-no-leads windows are
    # meaningfully penalised.  Distinct from leakage (which requires
    # confirmed leads + above-target CPA to fire).
    no_conversion_penalty = KAPPA * spend if leads == 0 and spend > 0 else 0.0

    return volume_score - cpa_penalty - leakage_penalty - no_conversion_penalty


# ---------------------------------------------------------------------------
# Pacing Score: shadow-price reward for budget utilisation
# ---------------------------------------------------------------------------

def calculate_pacing_score(window_df: pd.DataFrame) -> float:
    """
    Pacing score using the Lagrangian shadow-price interpretation.

    F = U + Pacing Score where:

        Pacing Score = λ · min(spend, target)
                     − OVERSPEND_FACTOR · λ · max(spend − target, 0)

    Rationale
    ---------
    λ = exp(t/24) is the shadow price of budget utilisation: it grows
    through the day, increasing the value (and urgency) of deploying budget.

    • Spending up to the volatility-adjusted hourly target is *rewarded*
      proportional to λ — deploying budget earns value.
    • Overshooting the target is penalised at OVERSPEND_FACTOR × λ — the
      asymmetry (2×) discourages exceeding the pacing target.
    • The hourly target is volatility-adjusted: during a website crash
      (volatility = 0) the target drops to zero, so spending nothing incurs
      no underspend penalty while any spend *during* the crash is correctly
      flagged as over-target (toxic).

    Expected signs
    --------------
    Normal window (spend ≈ 0.5 × target, 0 leads) : F ≈ +30  (positive)
    Normal window (leads at target CPA)            : F ≈ +150 (strongly positive)
    Crash window — MAS paused (spend ≈ 0)          : F ≈   0  (neutral)
    Crash window — IB still spending               : F < 0    (toxic spend penalised)
    """
    spend = window_df["spend"].sum()
    volatility = window_df["volatility"].mean()
    daily_budget = window_df["daily_budget"].mean()

    # Volatility-adjusted hourly target: zero during crash, full during normal
    target_spend = (daily_budget / 24.0) * volatility

    t_norm = (window_df["current_hour"].mean() % 24) / 24.0
    lam = math.exp(t_norm)

    pacing_reward = lam * min(spend, target_spend)
    overspend_penalty = OVERSPEND_FACTOR * lam * max(spend - target_spend, 0.0)

    return pacing_reward - overspend_penalty


# ---------------------------------------------------------------------------
# Optimisation function: F = U + Pacing Score
# ---------------------------------------------------------------------------

def calculate_opti_function(
    window_df: pd.DataFrame,
    is_holiday: bool,
) -> float:
    """
    F = U + Pacing Score

    The pacing score embeds the shadow price λ = exp(t/24), so F reflects
    both the economic value created (leads, CPA efficiency) and the
    efficiency of budget deployment (pacing quality).

    Expected behaviour by scenario
    --------------------------------
    Normal, active, 0 leads   : F > 0   (pacing reward dominates)
    Normal, leads at target    : F >> 0  (both utility and pacing positive)
    Crash, system paused       : F ≈ 0   (target = 0, spend = 0, no penalty)
    Crash, system still spending: F < 0  (any spend exceeds zero target)
    Holiday, low CVR           : F > 0   (lower pacing reward, but still positive)
    """
    utility = calculate_utility(window_df, volume_bonus=1.0 if is_holiday else 0.0)
    pacing = calculate_pacing_score(window_df)
    return utility + pacing
