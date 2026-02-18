from __future__ import annotations


class WorldClock:
    """
    Canonical simulation clock shared by every component.

    Tracks elapsed *minutes* and derives hour / day from that single
    counter.  All other modules should read time from this object
    rather than computing it independently.

    Usage:
        clock = WorldClock(step_minutes=15)
        clock.current_minute   # 0
        clock.current_hour     # 0
        clock.current_day      # 1  (1-based)
        clock.tick()           # advances by step_minutes
    """

    def __init__(
        self,
        step_minutes: int = 15,
        minutes_per_day: int = 60 * 24,
    ) -> None:
        self.step_minutes = int(step_minutes)
        self.minutes_per_day = int(minutes_per_day)
        self._current_minute: int = 0

    # ------------------------------------------------------------------
    # Read-only time accessors
    # ------------------------------------------------------------------
    @property
    def current_minute(self) -> int:
        return self._current_minute

    @property
    def current_hour(self) -> int:
        return self._current_minute // 60

    @property
    def current_day(self) -> int:
        """1-based day index."""
        return (self._current_minute // self.minutes_per_day) + 1

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """Advance the clock by one step (``step_minutes`` minutes)."""
        self._current_minute += self.step_minutes

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"WorldClock(minute={self.current_minute}, "
            f"hour={self.current_hour}, day={self.current_day})"
        )
