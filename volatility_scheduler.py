
class VolatilityScheduler:
    """
    "Market weather" controller.
    Returns a volatility multiplier (ν) based on current virtual hour.
    """

    def get_v_multiplier(self, hour: int) -> float:
        hour = int(hour)
        
        # Scenario 1: Website Crash (Hours 288 - 432)
        if 288 <= hour <= 432:
            return {"v": 0.0, "event": "CRASH"}

        # Scenario 2: Holiday/Competitor Entry (Hours 468-624)
        if 468 <= hour <= 624:
            return {"v": 0.7, "event": "HOLIDAY"}

        return {"v": 0.95, "event": "NORMAL"}

