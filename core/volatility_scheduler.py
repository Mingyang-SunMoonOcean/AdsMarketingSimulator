WEBSITE_CRASH_HOURS = [288, 432]
HOLIDAY_HOURS = [468, 624]

class VolatilityScheduler:
    """
    "Market weather" controller.
    Returns a volatility multiplier (ν) based on current virtual hour.
    """

    def get_v_multiplier(self, hour: int) -> float:
        hour = int(hour)
        
        # Scenario 1: Website Crash (Hours 288 - 432)
        if WEBSITE_CRASH_HOURS[0] <= hour <= WEBSITE_CRASH_HOURS[1]:
            return {"v": 0.0, "event": "CRASH"}

        # Scenario 2: Holiday/Competitor Entry (Hours 468-624)
        if HOLIDAY_HOURS[0] <= hour <= HOLIDAY_HOURS[1]:
            return {"v": 0.7, "event": "HOLIDAY"}

        return {"v": 1.0, "event": "NORMAL"}
