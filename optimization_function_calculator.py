import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import volatility_scheduler
import math

# --- 1. CONFIGURATION ---
ALPHA = 500.0  # Value per lead (CHF) (w1)
BONUS_VALUE_PER_LEAD = 100.0 # Strategic decision bonus per lead
BETA = 5.0     # Penalty per CHF deviation from Target CPA (w2)
GAMMA = 1.5    # Multiplier for Capital Leakage (toxic spend) (w3)
TARGET_CPA = 80.0
BASE_CVR = 0.025
WINDOW_SIZE = 4 #  hours (4 steps * 15 mins)

# --- 2. LOGIC ---
# U = volume_score - cpa_penalty - leakage_penalty
def calculate_utility(window_df, volume_bonus=0.0):

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

def calculate_budget_deviation(window_df):
    spend = window_df['spend'].sum()
    budget = window_df['daily_budget'].mean() / 24.0 # daily budget per hour

    budget_deviation = abs(spend - budget)
    return budget_deviation

def calculate_opti_function(window_df, is_holiday: bool) -> float:
    """F = U - λ·penalty, where λ (shadow price) = exp(t_norm)."""
    utility = calculate_utility(window_df, volume_bonus=1.0 if is_holiday else 0.0)
    penalty = calculate_budget_deviation(window_df)
    t_norm = (window_df['current_hour'].mean() % 24) / 24.0
    shadow_price = math.exp(t_norm)
    return utility - penalty * shadow_price

# --- 3. MAIN EXECUTION ---
def main():
    csv_file = 'ib_simulation_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run simulation first.")
        return

    df = pd.read_csv(csv_file)
    opti_results = []

    website_crash_hours = volatility_scheduler.WEBSITE_CRASH_HOURS
    holiday_hours = volatility_scheduler.HOLIDAY_HOURS

    # Rolling Window: compute optimization function F = U - λ·penalty
    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i:i+WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break
        current_hour = int(window['current_hour'].mean())
        is_holiday = holiday_hours[0] <= current_hour <= holiday_hours[1]
        is_crash = website_crash_hours[0] <= current_hour <= website_crash_hours[1]

        opti_value = calculate_opti_function(window, is_holiday)

        opti_results.append({
            'hour': window['current_hour'].mean(),
            'opti_function': opti_value,
            'volatility': window['volatility'].mean(),
            'event': "CRASH" if is_crash else "HOLIDAY" if is_holiday else "NORMAL"
        })

    opti_df = pd.DataFrame(opti_results)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(opti_df['hour'], opti_df['opti_function'], label='Optimization Function $F$', color='#2c3e50')

    # Dynamic Shading for Events
    plt.axvspan(website_crash_hours[0], website_crash_hours[1], color='red', alpha=0.15, label='Website Crash')
    plt.axvspan(holiday_hours[0], holiday_hours[1], color='orange', alpha=0.15, label='Competitor Spike')

    plt.title('Optimization Function Analysis: Zurich Dealership Baseline', fontsize=14)
    plt.ylabel('Optimization Function Value ($F$)')
    plt.xlabel('Simulation Time (Hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('optimization_function_graph.png')
    opti_df.to_csv('optimization_function_results.csv', index=False)
    print("Success: optimization_function_graph.png and optimization_function_results.csv generated.")

if __name__ == "__main__":
    main()