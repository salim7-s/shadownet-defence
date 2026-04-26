"""Import and smoke tests for training/ helpers (no GPU)."""

from __future__ import annotations

from training.prompts import SYSTEM_PROMPT, build_observation_prompt, format_action_block
from training.reward_adapter import (
    EpisodeRunner,
    compute_format_reward,
    parse_action_output,
)
from environment import ShadowNetEnv
from models import SREAction, ActionType


def test_format_reward_respects_action_lines():
    text = "ACTION: wait_and_track\nTARGET: none\nREASON: test"
    assert compute_format_reward(text) >= 0.9


def test_parse_valid_wait():
    env = ShadowNetEnv()
    obs = env.reset("shadow-easy", scenario_index=0, seed=0)
    pr = parse_action_output(
        "ACTION: wait_and_track\nTARGET: none",
        obs,
    )
    assert pr.valid
    assert pr.action.action_type == ActionType.WAIT_AND_TRACK


def test_episode_runner_baseline_fn():
    """Deterministic 'policy' for unit test: always wait (valid everywhere)."""
    env = ShadowNetEnv()
    obs = env.reset("shadow-easy", scenario_index=0, seed=0)

    def policy_fn(_sys: str, _user: str, o) -> str:
        return format_action_block(
            SREAction(action_type=ActionType.WAIT_AND_TRACK, target=None),
            reason="test",
        )

    runner = EpisodeRunner()
    out = runner.run_episode(
        policy_fn,
        "shadow-easy",
        seed=0,
        scenario_index=0,
        attacker_profile="stealthy",
    )
    assert "score" in out
    assert out["steps"] >= 1
    assert 0.01 < out["score"] < 1.0
