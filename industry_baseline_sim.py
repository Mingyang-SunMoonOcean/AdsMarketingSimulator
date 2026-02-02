from __future__ import annotations

from typing import Tuple

from sandbox_env import SandboxEnv


def run_industry_baseline_simulation(
    total_steps: int = 2880,
    hourly_step_interval: int = 4,
    human_step_interval: int = 48,
    target_cpa: float = 80.0,
    initial_max_bid: float = 5.00,
    initial_daily_budget: float = 1000.0,
) -> Tuple[dict, dict]:
    """
    Run the 30‑day (2,880‑step) "Industry Baseline" simulation.

    Two control loops operate on the same SandboxEnv instance:
    - Loop A (Proportional Rule): every virtual hour (every 4 steps).
    - Loop B (Human Intervener): every 24 virtual hours (every 96 steps).

    Returns a tuple of (final_state, aggregate_metrics).
    """
    env = SandboxEnv()

    # Initial configuration
    env.configure(daily_budget=initial_daily_budget, max_bid=initial_max_bid)

    # Aggregates for reporting
    total_spend = 0.0
    total_leads = 0
    total_clicks = 0

    for step in range(total_steps):
        # Advance simulation by one 15‑minute step
        outcome = env.act()

        # Update aggregates
        total_spend += float(outcome.get("spend", 0.0))
        total_leads += int(outcome.get("leads", 0))
        total_clicks += int(outcome.get("clicks", 0))

        # Observe current state (includes latest_outcome and config)
        obs = env.observe()
        latest = obs.get("latest_outcome") or {}

        # ------------------------------------------------------------------
        # Loop A: Proportional Rule (Automation) – every 1 virtual hour
        # ------------------------------------------------------------------
        if (step + 1) % hourly_step_interval == 0:
            leads = int(latest.get("leads", 0))

            # Edge case: no leads → cannot compute CPA; status quo
            if leads > 0:
                cpa = float(latest.get("cpa", 0.0))
                max_bid = float(obs.get("max_bid", 0.0))

                if cpa > target_cpa:
                    # CPA above target → reduce max_bid by 5%
                    new_max_bid = max_bid * 0.95
                    env.configure(max_bid=new_max_bid)
                elif cpa < target_cpa * 0.8:
                    # CPA comfortably below 80% of target → increase max_bid by 2%
                    new_max_bid = max_bid * 1.02
                    env.configure(max_bid=new_max_bid)
                # Else: within band → no change

        # ------------------------------------------------------------------
        # Loop B: Human Intervener (Manual) – every 12 virtual hours
        # ------------------------------------------------------------------
        if (step + 1) % human_step_interval == 0:
            # Look back over the last 24 hours (96 steps) of history.
            history = env.state.state.history
            window = history[-human_step_interval:] if len(history) >= 1 else []

            spend_24h = sum(float(r.get("spend", 0.0)) for r in window)
            leads_24h = sum(int(r.get("leads", 0)) for r in window)

            # If money is being spent but no leads are coming in, assume a crash.
            if spend_24h > 0.0 and leads_24h == 0:
                # Emergency pause
                env.configure(max_bid=0.01)

    final_state = env.observe()
    aggregate_metrics = {
        "total_spend": round(total_spend, 4),
        "total_leads": int(total_leads),
        "total_clicks": int(total_clicks),
        "overall_cpa": round(total_spend / total_leads, 4) if total_leads > 0 else None,
    }

    return final_state, aggregate_metrics


if __name__ == "__main__":
    final_state, metrics = run_industry_baseline_simulation()

    print("=== Industry Baseline Simulation (30 days) ===")
    print(f"Final virtual day: {final_state.get('day')}")
    print(f"Final virtual hour: {final_state.get('hour')}")
    print(f"Final max_bid: {final_state.get('max_bid')}")
    print(f"Final daily_budget: {final_state.get('daily_budget')}")
    print("--- Aggregates ---")
    print(f"Total spend: {metrics['total_spend']}")
    print(f"Total leads: {metrics['total_leads']}")
    print(f"Total clicks: {metrics['total_clicks']}")
    print(f"Overall CPA: {metrics['overall_cpa']}")

