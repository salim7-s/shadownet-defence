"""
ShadowNet — Baseline Rule-Based Agent
═══════════════════════════════════════════════════════════════
Deterministic strategy with NO ML.

Measured baseline scores (5 scenarios each, seed=42):
  shadow-easy:   ~0.52–0.59  (avg ~0.55)
  shadow-medium: ~0.47–0.50  (avg ~0.49)
  shadow-hard:   ~0.45–0.47  (avg ~0.46)

Used as the "before" in the before/after comparison table.
Any trained policy should materially exceed these numbers.

Strategy:
  TRACK:    Observe every anomalous node systematically
  CONTAIN:  Mirror before redirect, space actions out
  EVIDENCE: Lock artifacts by decay rate (fastest expiry first)
  SAFETY:   If attacker_query_frequency > 10, wait_and_track
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import List, Optional
from models import SREAction, SREObservation, ActionType


class BaselineAgent:
    """
    Rule-based baseline agent for ShadowNet.
    No learning — pure heuristic. Shows what score is achievable
    without any training, establishing the improvement baseline.
    """

    def __init__(self):
        self._observed_nodes: List[str] = []
        self._mirrored_nodes: List[str] = []
        self._redirected_nodes: List[str] = []
        self._nodes_to_process: List[str] = []

    def reset(self):
        """Reset agent state for new episode."""
        self._observed_nodes = []
        self._mirrored_nodes = []
        self._redirected_nodes = []
        self._nodes_to_process = []

    def act(self, obs: SREObservation) -> SREAction:
        """
        Choose an action based on current observation.
        Returns SREAction.
        """
        # Safety rule: if attacker is getting suspicious, back off
        if obs.attacker_query_frequency > 10:
            return SREAction(action_type=ActionType.WAIT_AND_TRACK)

        if obs.phase == "track":
            return self._act_track(obs)
        elif obs.phase == "contain":
            return self._act_contain(obs)
        elif obs.phase == "evidence":
            return self._act_evidence(obs)
        else:
            return SREAction(action_type=ActionType.WAIT_AND_TRACK)

    def _act_track(self, obs: SREObservation) -> SREAction:
        """Phase 1: Proactively observe nodes to discover attacker activity."""
        # Systematically observe all network nodes to find attacker
        unobserved = [n for n in obs.network_nodes if n not in obs.visible_nodes]
        if unobserved:
            return SREAction(action_type=ActionType.OBSERVE, target=unobserved[0])

        # If we have anomalous nodes, mirror them for containment prep
        unmirrored = [n for n in obs.anomalous_nodes if n not in self._mirrored_nodes]
        if unmirrored:
            target = unmirrored[0]
            self._mirrored_nodes.append(target)
            return SREAction(action_type=ActionType.MIRROR_TRAFFIC, target=target)

        # If we've mirrored anomalous nodes, start redirecting (transition to contain)
        ready = [n for n in obs.mirrored_nodes if n not in obs.honeypot_nodes]
        if ready:
            return SREAction(action_type=ActionType.REDIRECT, target=ready[0])

        # Default: wait and gather passive intel
        return SREAction(action_type=ActionType.WAIT_AND_TRACK)

    def _act_contain(self, obs: SREObservation) -> SREAction:
        """Phase 2: Mirror first, then redirect. Keep scanning for attacker."""
        # First priority: keep observing unscanned nodes to discover attacker
        unobserved = [n for n in obs.network_nodes if n not in obs.visible_nodes]
        if unobserved and len(obs.anomalous_nodes) < 2:
            return SREAction(action_type=ActionType.OBSERVE, target=unobserved[0])

        # If last action was flagged, back off
        if obs.last_action_outcome in ("flagged", "minor_anomaly"):
            return SREAction(action_type=ActionType.WAIT_AND_TRACK)

        # Mirror anomalous nodes before redirecting
        anomalous = obs.anomalous_nodes
        unmirrored = [n for n in anomalous
                      if n not in obs.mirrored_nodes and n not in obs.honeypot_nodes]
        if unmirrored:
            target = unmirrored[0]
            self._mirrored_nodes.append(target)
            return SREAction(action_type=ActionType.MIRROR_TRAFFIC, target=target)

        # Redirect mirrored nodes
        ready_to_redirect = [n for n in obs.mirrored_nodes
                             if n not in obs.honeypot_nodes and n not in self._redirected_nodes]
        if ready_to_redirect:
            target = ready_to_redirect[0]
            self._redirected_nodes.append(target)
            return SREAction(action_type=ActionType.REDIRECT, target=ready_to_redirect[0])

        # Still unobserved nodes? Keep scanning
        if unobserved:
            return SREAction(action_type=ActionType.OBSERVE, target=unobserved[0])

        return SREAction(action_type=ActionType.WAIT_AND_TRACK)

    def _act_evidence(self, obs: SREObservation) -> SREAction:
        """Phase 3: Lock artifacts by decay rate (fastest expiry first)."""
        if not obs.available_artifacts:
            return SREAction(action_type=ActionType.WAIT_AND_TRACK)

        # Sort by steps_remaining (lock most urgent first)
        sorted_arts = sorted(obs.available_artifacts, key=lambda a: a.get("steps_remaining", 99))

        for art in sorted_arts:
            art_id = art["id"]
            if art_id not in obs.secured_artifacts:
                return SREAction(action_type=ActionType.LOCK_ARTIFACT, target=art_id)

        return SREAction(action_type=ActionType.WAIT_AND_TRACK)


def run_baseline_episode(env, task_name: str = "shadow-easy") -> dict:
    """Run one full episode with the baseline agent. Returns grade dict."""
    from grader import grade_episode

    agent = BaselineAgent()
    agent.reset()

    obs = env.reset(task_name)

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)

    return grade_episode(env.state)


def run_baseline_all_tasks(env) -> dict:
    """Run baseline agent across all tasks, return aggregate results."""
    results = {}
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]

    for task in tasks:
        scores = []
        for i in range(5):  # run all 5 scenarios per difficulty
            env_copy = type(env)()
            obs = env_copy.reset(task, scenario_index=i)
            agent = BaselineAgent()
            agent.reset()

            while not obs.done:
                action = agent.act(obs)
                obs = env_copy.step(action)

            from grader import grade_episode
            grade = grade_episode(env_copy.state)
            scores.append(grade["score"])

        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = {
            "avg_score": round(avg, 3),
            "scores": scores,
            "episodes_run": len(scores),
        }

    return results


# ─── RANDOM AGENT ─────────────────────────────────────────────────────────────

class RandomAgent:
    """
    Uniformly random agent — picks any valid action at each step.
    Used as the "floor" benchmark: any trained / rule-based agent should beat this.
    """

    import random as _random

    def act(self, obs: SREObservation) -> SREAction:
        """Choose a random action from the valid actions list."""
        import random
        if obs.valid_actions:
            action_str = random.choice(obs.valid_actions)
        else:
            action_str = "wait_and_track"

        action_map = {
            "observe":          ActionType.OBSERVE,
            "mirror_traffic":   ActionType.MIRROR_TRAFFIC,
            "redirect":         ActionType.REDIRECT,
            "loud_contain":     ActionType.LOUD_CONTAIN,
            "partial_covert":   ActionType.PARTIAL_COVERT,
            "wait_and_track":   ActionType.WAIT_AND_TRACK,
            "lock_artifact":    ActionType.LOCK_ARTIFACT,
            "emergency_expel":  ActionType.EMERGENCY_EXPEL,
        }
        action_type = action_map.get(action_str, ActionType.WAIT_AND_TRACK)

        # For actions that need a target, pick a random node / artifact
        target = None
        if action_type in (ActionType.OBSERVE, ActionType.MIRROR_TRAFFIC,
                           ActionType.REDIRECT, ActionType.LOUD_CONTAIN,
                           ActionType.PARTIAL_COVERT):
            if obs.network_nodes:
                import random
                target = random.choice(obs.network_nodes)
        elif action_type == ActionType.LOCK_ARTIFACT:
            if obs.available_artifacts:
                import random
                target = random.choice(obs.available_artifacts)["id"]

        return SREAction(action_type=action_type, target=target)


def run_random_episode(env, task_name: str = "shadow-easy") -> dict:
    """Run one full episode with the random agent. Returns grade dict."""
    from grader import grade_episode

    agent = RandomAgent()
    obs = env.reset(task_name)

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)

    return grade_episode(env.state)


def run_random_all_tasks(env) -> dict:
    """Run random agent across all tasks, return aggregate results."""
    results = {}
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]

    for task in tasks:
        scores = []
        for i in range(5):
            env_copy = type(env)()
            obs = env_copy.reset(task, scenario_index=i)
            agent = RandomAgent()

            while not obs.done:
                action = agent.act(obs)
                obs = env_copy.step(action)

            from grader import grade_episode
            grade = grade_episode(env_copy.state)
            scores.append(grade["score"])

        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = {
            "avg_score": round(avg, 3),
            "scores": scores,
            "episodes_run": len(scores),
        }

    return results
