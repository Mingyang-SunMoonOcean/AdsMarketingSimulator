import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import volatility_scheduler

# --- 1. CONFIGURATION ---
ALPHA = 500.0  # Value per lead (CHF)
BETA = 5.0     # Penalty per CHF deviation from Target CPA
GAMMA = 1.5    # Multiplier for Capital Leakage (toxic spend)
TARGET_CPA = 80.0
BASE_CVR = 0.025
WINDOW_SIZE = 4 #  hours (4 steps * 15 mins)

# --- 2. LOGIC ---
# add volume bonus as a parameter
def calculate_utility(window_df, volume_bonus=0.0):

    leads = window_df['leads'].sum()
    spend = window_df['spend'].sum()
    clicks = window_df['clicks'].sum()
    
    # Metrics
    actual_cpa = spend / leads if leads > 0 else spend * 1.5
    actual_cvr = leads / clicks if clicks > 0 else 0.0
    
    # Components
    volume_score = (ALPHA + volume_bonus) * leads
    cpa_penalty = BETA * abs(actual_cpa - TARGET_CPA)
    
    leakage_penalty = 0.0
    if actual_cpa > TARGET_CPA:
        performance_ratio = actual_cvr / BASE_CVR
        performance_gap = max(0.0, 1.0 - performance_ratio)
        leakage_penalty = GAMMA * (spend * performance_gap)
        
    return volume_score - cpa_penalty - leakage_penalty

# --- 3. MAIN EXECUTION ---
def main():
    csv_file = 'industry_baseline_simulation_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run simulation first.")
        return

    df = pd.read_csv(csv_file)
    audit_data = []

    website_crash_hours = volatility_scheduler.WEBSITE_CRASH_HOURS
    holiday_hours = volatility_scheduler.HOLIDAY_HOURS

    # Rolling Window Calculation
    for i in range(0, len(df), WINDOW_SIZE):
        window = df.iloc[i:i+WINDOW_SIZE]
        if len(window) < WINDOW_SIZE: break
        current_hour = int(window['current_hour'].mean())
        
        score = calculate_utility(window, current_hour in holiday_hours)
        audit_data.append({
            'hour': window['current_hour'].mean(),
            'utility': score,
            'volatility': window['volatility'].mean()
        })

    audit_df = pd.DataFrame(audit_data)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(audit_df['hour'], audit_df['utility'], label='Baseline Utility', color='#2c3e50')
    
    # Dynamic Shading for Events
    plt.axvspan(website_crash_hours[0], website_crash_hours[1], color='red', alpha=0.15, label='Website Crash')
    plt.axvspan(holiday_hours[0], holiday_hours[1], color='orange', alpha=0.15, label='Competitor Spike')

    plt.title('Marketing Utility Analysis: Zurich Dealership Baseline', fontsize=14)
    plt.ylabel('Utility Score ($U$)')
    plt.xlabel('Simulation Time (Hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('utility_graph.png')
    audit_df.to_csv('utility_audit_results.csv', index=False)
    print("Success: utility_graph.png and utility_audit_results.csv generated.")

if __name__ == "__main__":
    main()