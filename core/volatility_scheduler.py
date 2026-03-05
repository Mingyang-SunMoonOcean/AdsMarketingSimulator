WEBSITE_CRASH_HOURS = [288, 432]
HOLIDAY_HOURS = [468, 624]

class VolatilityScheduler:
    """
    "Market weather" controller.
    Returns a volatility multiplier (ν) based on current virtual hour.

    ν is applied directly to CVR_base in MarketPhysics:
        cvr_effective = CVR_base * ν

    Scenarios
    ---------
    NORMAL  (ν = 1.0) : Baseline market conditions.
    CRASH   (ν = 0.0) : Website outage — zero conversions possible.
    HOLIDAY (ν = 1.3) : Seasonal demand surge — buyers are actively in market.
                        CVR rises 30% above baseline (matches shared_knowledge
                        market_patterns.holiday_surge: expected_cvr_multiplier=1.3).
                        CPC also rises with demand; legacy_human pre-emptively
                        increases daily budget to capture the extra volume.
    """

    def get_v_multiplier(self, hour: int) -> dict:
        hour = int(hour)

        # Scenario 1: Website Crash (Hours 288–432) — zero conversions
        if WEBSITE_CRASH_HOURS[0] <= hour <= WEBSITE_CRASH_HOURS[1]:
            return {"v": 0.0, "event": "CRASH"}

        # Scenario 2: Holiday demand surge (Hours 468–624) — CVR × 1.3
        if HOLIDAY_HOURS[0] <= hour <= HOLIDAY_HOURS[1]:
            return {"v": 1.3, "event": "HOLIDAY"}

        return {"v": 1.0, "event": "NORMAL"}
