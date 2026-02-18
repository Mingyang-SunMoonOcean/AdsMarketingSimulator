"""API interface to Sandbox for OODA MAS agents."""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from core.sandbox_env import SandboxEnv


class ConfigInput(BaseModel):
    daily_budget: Optional[float] = None
    max_bid: Optional[float] = None


def create_app(env: Optional[SandboxEnv] = None) -> FastAPI:
    """
    FastAPI surface for operating agents.

    - POST /config: write daily budget / max bid into StateManager.
    - POST /act: advance one 15‑minute step and return the latest outcome.
    - GET  /observe: read the latest observable state (including last outcome).
    """
    app = FastAPI()
    env = env or SandboxEnv()

    @app.get("/observe")
    def observe():
        """Agents call this to see current state and latest outcome."""
        return env.observe()

    @app.post("/config")
    def config(data: ConfigInput):
        """Agents call this to set daily budget and/or max bid."""
        env.configure(daily_budget=data.daily_budget, max_bid=data.max_bid)
        return {"status": "ok"}

    @app.post("/act")
    def act():
        """Simulation script call this to advance time by one step."""
        return env.act()

    return app


class Executor:
    """
    Executor agent: interfaces with the Sandbox via API or direct calls.

    In simulation mode, holds a reference to SandboxEnv and calls
    configure() / observe() directly. In API mode, makes HTTP requests.
    """

    def __init__(self, env: SandboxEnv):
        self.env = env

    def observe(self):
        """Observe current state from the sandbox."""
        return self.env.observe()

    def configure(self, daily_budget: Optional[float] = None, max_bid: Optional[float] = None):
        """Write configuration to the sandbox."""
        self.env.configure(daily_budget=daily_budget, max_bid=max_bid)

    def act(self):
        """Advance simulation by one step (typically called by simulation driver)."""
        return self.env.act()
