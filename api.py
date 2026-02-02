from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from sandbox_env import SandboxEnv


class ConfigInput(BaseModel):
    daily_budget: float | None = None
    max_bid: float | None = None


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