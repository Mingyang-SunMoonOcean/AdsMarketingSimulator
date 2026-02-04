
class VolatilityScheduler:
    """
    "Market weather" controller.
    Returns a volatility multiplier (ν) based on current virtual hour.
    """

    def get_v_multiplier(self, hour: int) -> float:
        hour = int(hour)
        
        # Scenario 1: Website Crash (Hours 120-168)
        if 120 <= hour <= 168:
            return {"v": 0.0, "event": "CRASH"}

        # Scenario 2: Holiday/Competitor Entry (Hours 300-468)
        if 300 <= hour <= 468:
            return {"v": 0.7, "event": "HOLIDAY"}

        return {"v": 0.95, "event": "NORMAL"}

