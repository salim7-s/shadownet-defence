"""
Rollout and reward adapter utilities for ShadowNet training/evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from environment import ShadowNetEnv, ATTACKER_PROFILES
from grader import grade_episode
from models import ActionType, SREAction, SREObservation
from training.prompts import SYSTEM_PROMPT, build_observation_prompt


TARGETED_ACTIONS = {
    ActionType.OBSERVE,
    ActionType.MIRROR_TRAFFIC,
    ActionType.REDIRECT,
    ActionType.LOUD_CONTAIN,
    ActionType.PARTIAL_COVERT,
    ActionType.LOCK_ARTIFACT,
}


ACTION_MAP = {
    "observe": ActionType.OBSERVE,
    "mirror_traffic": ActionType.MIRROR_TRAFFIC,
    "redirect": ActionType.REDIRECT,
    "loud_contain": ActionType.LOUD_CONTAIN,
    "partial_covert": ActionType.PARTIAL_COVERT,
    "wait_and_track": ActionType.WAIT_AND_TRACK,
    "lock_artifact": ActionType.LOCK_ARTIFACT,
    "emergency_expel": ActionType.EMERGENCY_EXPEL,
}


@dataclass
class ParseResult:
    action: SREAction
    valid: bool
    error: str | None = None
    format_reward: float = 0.0
    output_penalty: float = 0.0


def compute_format_reward(text: str) -> float:
    """Reward structured output without over-weighting it."""
    upper = text.upper()
    score = 0.0
    if "ACTION:" in upper:
        score += 0.7
    if "TARGET:" in upper:
        score += 0.2
    if "REASON:" in upper:
        score += 0.1
    return min(score, 1.0)


def parse_action_output(text: str, obs: SREObservation) -> ParseResult:
    """Parse model output into an action, falling back safely on invalid outputs."""
    format_reward = compute_format_reward(text)
    match = re.search(r"ACTION:\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
    if not match:
        return ParseResult(
            action=SREAction(action_type=ActionType.WAIT_AND_TRACK),
            valid=False,
            error="missing_action",
            format_reward=format_reward,
            output_penalty=-0.10,
        )

    action_name = match.group(1).strip().lower()
    action_type = ACTION_MAP.get(action_name)
    if action_type is None or action_name not in obs.valid_actions:
        return ParseResult(
            action=SREAction(action_type=ActionType.WAIT_AND_TRACK),
            valid=False,
            error="invalid_action",
            format_reward=format_reward,
            output_penalty=-0.08,
        )

    target_match = re.search(r"TARGET:\s*(.+)", text, flags=re.IGNORECASE)
    raw_target = target_match.group(1).strip() if target_match else ""
    target = None if raw_target.lower() in {"", "none", "null"} else raw_target

    if action_type in TARGETED_ACTIONS:
        valid_targets = []
        if action_type == ActionType.LOCK_ARTIFACT:
            valid_targets = [a["id"] for a in obs.available_artifacts]
        else:
            valid_targets = list(obs.network_nodes)
        if target not in valid_targets:
            return ParseResult(
                action=SREAction(action_type=ActionType.WAIT_AND_TRACK),
                valid=False,
                error="invalid_target",
                format_reward=format_reward,
                output_penalty=-0.08,
            )

    return ParseResult(
        action=SREAction(action_type=action_type, target=target),
        valid=True,
        format_reward=format_reward,
        output_penalty=0.0,
    )


class EpisodeRunner:
    """Run a full ShadowNet episode from a text-generating policy."""

    def __init__(self, env_factory: Callable[[], ShadowNetEnv] = ShadowNetEnv):
        self._env_factory = env_factory

    def run_episode(
        self,
        policy_fn: Callable[[str, str, SREObservation], str],
        task_name: str,
        *,
        seed: int | None = None,
        scenario_index: int | None = None,
        attacker_profile: str | None = None,
    ) -> dict[str, Any]:
        env = self._env_factory()
        obs = env.reset(task_name, scenario_index=scenario_index, seed=seed)
        if attacker_profile in ATTACKER_PROFILES:
            env._attacker_profile_name = attacker_profile  # type: ignore[attr-defined]
            env._attacker_profile = ATTACKER_PROFILES[attacker_profile]  # type: ignore[attr-defined]

        trajectory: list[dict[str, Any]] = []
        parse_failures = 0
        format_rewards: list[float] = []
        penalties: list[float] = []

        while not obs.done:
            user_prompt = build_observation_prompt(obs)
            step_before = obs.step
            phase_before = obs.phase
            behavior_before = obs.attacker_behavior
            completion = policy_fn(SYSTEM_PROMPT, user_prompt, obs)
            parsed = parse_action_output(completion, obs)
            if not parsed.valid:
                parse_failures += 1
            format_rewards.append(parsed.format_reward)
            penalties.append(parsed.output_penalty)

            next_obs = env.step(parsed.action)
            trajectory.append(
                {
                    "step": step_before,
                    "prompt": user_prompt,
                    "completion": completion,
                    "parsed_action": parsed.action.model_dump(),
                    "parse_valid": parsed.valid,
                    "parse_error": parsed.error,
                    "phase": phase_before,
                    "attacker_behavior": behavior_before,
                }
            )
            obs = next_obs

        grade = grade_episode(env.state)
        avg_format_reward = sum(format_rewards) / max(len(format_rewards), 1)
        avg_penalty = sum(penalties) / max(len(penalties), 1)
        combined_reward = (grade["score"] * 0.70) + (avg_format_reward * 0.25) + avg_penalty

        return {
            "task": task_name,
            "seed": seed,
            "scenario_index": scenario_index,
            "attacker_profile": attacker_profile or env._attacker_profile_name,  # type: ignore[attr-defined]
            "score": grade["score"],
            "combined_reward": round(combined_reward, 4),
            "format_reward": round(avg_format_reward, 4),
            "output_penalty": round(avg_penalty, 4),
            "parse_failures": parse_failures,
            "steps": len(trajectory),
            "collapse": env.state.attacker_alerted,
            "grade": grade,
            "trajectory": trajectory,
        }
