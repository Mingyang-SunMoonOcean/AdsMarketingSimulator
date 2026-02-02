
class VolatilityScheduler:
    """
    "Market weather" controller.
    Returns a volatility multiplier (ν) based on current virtual hour.
    """

    def get_v_multiplier(self, hour: int) -> float:
        hour = int(hour)

        # Scenario 1: Website Crash (Hours 120-132)
        if 120 <= hour <= 132:
            return 0.0  # Zero conversions

        # Scenario 2: Competitor Entry (Hour 300 - 468, holiday season) 
        if 300 <= hour <= 468:
            return 0.7  # 30% drop in efficiency due to competition

        return 1.0  # Normal conditions

