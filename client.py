"""
ShadowNet — OpenEnv Client
═══════════════════════════════════════════════════════════════
Client for connecting to a running ShadowNet environment server.
Uses OpenEnv's GenericEnvClient for WebSocket-based communication.

Usage:
    # Async
    async with ShadowNetClient(base_url="https://YOUR-SPACE.hf.space") as client:
        result = await client.reset()
        result = await client.step(SREAction(action_type=ActionType.OBSERVE, target="auth-service"))

    # Sync
    with ShadowNetClient(base_url="https://YOUR-SPACE.hf.space").sync() as client:
        result = client.reset()
        result = client.step(SREAction(action_type=ActionType.OBSERVE, target="auth-service"))
"""

from __future__ import annotations

from models import SREAction, SREObservation

try:
    from openenv import GenericEnvClient

    class ShadowNetClient(GenericEnvClient[SREAction, SREObservation]):
        """OpenEnv client for ShadowNet environment."""
        ACTION_CLS = SREAction
        OBSERVATION_CLS = SREObservation

except ImportError:
    # Fallback: simple HTTP client when openenv not installed
    import requests
    from models import ActionType

    class ShadowNetClient:
        """Simple HTTP client fallback for ShadowNet."""

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self, task_name: str = "shadow-easy") -> dict:
            r = requests.post(f"{self.base_url}/reset", json={"task_name": task_name})
            return r.json()

        def step(self, action: SREAction) -> dict:
            r = requests.post(f"{self.base_url}/step", json={
                "action": {"action_type": action.action_type.value, "target": action.target}
            })
            return r.json()

        def state(self) -> dict:
            return requests.get(f"{self.base_url}/state").json()

        def health(self) -> dict:
            return requests.get(f"{self.base_url}/health").json()
