"""Total Optimization Function (U - P) for OODA MAS."""

import pandas as pd
import math


# --- Configuration ---
ALPHA = 500.0  # Value per lead (CHF) (w1)
BONUS_VALUE_PER_LEAD = 100.0  # Strategic decision bonus per lead
BETA = 5.0     # Penalty per CHF deviation from Target CPA (w2)
GAMMA = 1.5    # Multiplier for Capital Leakage (toxic spend) (w3)
TARGET_CPA = 80.0
BASE_CVR = 0.025


def calculate_utility(
    window_df: pd.DataFrame,
    volume_bonus: float = 0.0,
) -> float:
    """
    U = volume_score - cpa_penalty - leakage_penalty
    """
    leads = window_df['leads'].sum()
    spend = window_df['spend'].sum()
    clicks = window_df['clicks'].sum()

    # Metrics
    actual_cpa = spend / leads if leads > 0 else spend * 1.5
    actual_cvr = leads / clicks if clicks > 0 else 0.0

    # Components
    volume_score = (ALPHA + volume_bonus * BONUS_VALUE_PER_LEAD) * leads
    cpa_penalty = BETA * abs(actual_cpa - TARGET_CPA)

    leakage_penalty = 0.0
    if actual_cpa > TARGET_CPA:
        performance_ratio = actual_cvr / BASE_CVR
        performance_gap = max(0.0, 1.0 - performance_ratio)
        leakage_penalty = GAMMA * (spend * performance_gap)

    return volume_score - cpa_penalty - leakage_penalty


def calculate_budget_deviation(window_df: pd.DataFrame) -> float:
    """Budget deviation penalty for the optimization function."""
    spend = window_df['spend'].sum()
    budget = window_df['daily_budget'].mean() / 24.0  # daily budget per hour
    return abs(spend - budget)


def calculate_opti_function(
    window_df: pd.DataFrame,
    is_holiday: bool,
) -> float:
    """
    F = U - λ·penalty, where λ (shadow price) = exp(t_norm).
    """
    utility = calculate_utility(window_df, volume_bonus=1.0 if is_holiday else 0.0)
    penalty = calculate_budget_deviation(window_df)
    t_norm = (window_df['current_hour'].mean() % 24) / 24.0
    lam = math.exp(t_norm)
    return utility - penalty * lam
