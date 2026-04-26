"""
ShadowNet — LLM Inference Agent
═══════════════════════════════════════════════════════════════
Structured [START][STEP][END] logging (OpenEnv requirement).
Chain-of-thought reasoning with OBSERVED/INFERRED/ACTION/REASON.
Parseable action output for training loop integration.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional

from models import SREAction, SREObservation, ActionType
from environment import ShadowNetEnv
from grader import compute_reward, grade_episode

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Threat Containment Specialist operating inside ShadowNet.
Your mission: contain an active network intruder WITHOUT alerting them.

You cannot see the attacker's suspicion level. You must INFER it from:
- attacker_query_frequency (higher = more suspicious)
- honeypot_probe_count (higher = attacker testing consistency)
- attacker_lateral_attempts (higher = suspicious movement)
- last_action_outcome (smooth/minor_anomaly/flagged/alerted)
- attacker_behavior (passive/cautious/aggressive/alerted)

At each step, output your reasoning in this exact format:

OBSERVED: [what you see in the current observation]
INFERRED: [what you conclude about attacker suspicion level]
ACTION: [action_type] [target]
REASON: [why this action now]

Available actions by phase:
TRACK: observe <node>, mirror_traffic <node>, wait_and_track, emergency_expel
CONTAIN: observe <node>, mirror_traffic <node>, redirect <node>, loud_contain <node>, partial_covert <node>, wait_and_track, emergency_expel
EVIDENCE: lock_artifact <artifact_id>, wait_and_track, emergency_expel

Rules:
- ALWAYS mirror_traffic a node BEFORE redirecting it (reduces detection risk)
- If attacker_query_frequency > 10, consider wait_and_track to let suspicion drop
- In EVIDENCE phase, lock artifacts with lowest steps_remaining first
- emergency_expel ends everything — score ~0.25. Only use as last resort.
"""


def format_observation(obs: SREObservation) -> str:
    """Format observation into a readable prompt for the LLM."""
    lines = [
        f"Step: {obs.step}/{obs.max_steps} | Phase: {obs.phase}",
        f"Anomalous nodes: {obs.anomalous_nodes}",
        f"Honeypot nodes: {obs.honeypot_nodes}",
        f"Mirrored nodes: {obs.mirrored_nodes}",
        f"Contained nodes: {obs.contained_nodes}",
        f"attacker_query_frequency: {obs.attacker_query_frequency}",
        f"honeypot_probe_count: {obs.honeypot_probe_count}",
        f"attacker_lateral_attempts: {obs.attacker_lateral_attempts}",
        f"last_action_outcome: {obs.last_action_outcome}",
        f"attacker_behavior: {obs.attacker_behavior}",
        f"Valid actions: {obs.valid_actions}",
    ]
    if obs.active_alerts:
        lines.append(f"ALERTS: {obs.active_alerts}")
    if obs.available_artifacts:
        arts = [f"{a['id']}(ttl={a['steps_remaining']})" for a in obs.available_artifacts]
        lines.append(f"Available artifacts: {arts}")
    if obs.secured_artifacts:
        lines.append(f"Secured artifacts: {obs.secured_artifacts}")

    return "\n".join(lines)


def format_structured_decision_log(obs: SREObservation, action: SREAction) -> str:
    """
    Structured OBSERVED / INFERRED / ACTION / REASON block.
    Kept in sync with the live server formatter (used for SFT labels + demos).
    """
    freq = obs.attacker_query_frequency
    probes = obs.honeypot_probe_count
    lateral = obs.attacker_lateral_attempts
    behavior = obs.attacker_behavior

    observed = (
        f"phase={obs.phase}, step={obs.step}/{obs.max_steps}, "
        f"behavior={behavior}, query_freq={freq}, probes={probes}, lateral={lateral}, "
        f"fidelity={obs.honeypot_fidelity:.2f}, outcome={obs.last_action_outcome}"
    )

    if behavior == "alerted":
        inferred = "Attacker went dark — logs purged, forensic window closing."
    elif freq >= 10 or probes >= 3:
        inferred = "Elevated signals. Attacker likely suspicious. Risk approaching threshold."
    elif freq >= 5 or lateral >= 2:
        inferred = "Moderate signals. Attacker cautious. Avoid aggressive actions."
    else:
        inferred = "Low signals. Attacker passive. Safe window to continue containment."

    action_str = f"{action.action_type.value}" + (f" {action.target}" if action.target else "")

    if action.action_type.value == "wait_and_track":
        reason = "Waiting to let detection risk decay before next containment move."
    elif action.action_type.value == "mirror_traffic":
        reason = "Mirroring to raise honeypot fidelity before redirect (reduces risk)."
    elif action.action_type.value == "redirect":
        reason = "Redirecting attacker into honeypot — fidelity high enough to avoid detection."
    elif action.action_type.value == "lock_artifact":
        reason = "Locking forensic artifact before it decays."
    elif action.action_type.value == "emergency_expel":
        reason = "Emergency expel — data at imminent risk, accepting evidence loss."
    else:
        reason = "Standard containment move based on current phase and signals."

    return (
        f"OBSERVED: {observed}\n"
        f"INFERRED: {inferred}\n"
        f"ACTION: {action_str}\n"
        f"REASON: {reason}"
    )


def parse_action(text: str) -> SREAction:
    """
    Parse LLM output to extract the ACTION line.
    Expected format: ACTION: action_type target
    Falls back to wait_and_track if parsing fails.
    """
    # Find ACTION: line
    match = re.search(r"ACTION:\s*(\S+)\s*(.*)", text, re.IGNORECASE)
    if not match:
        return SREAction(action_type=ActionType.WAIT_AND_TRACK)

    action_str = match.group(1).strip().lower()
    target_str = match.group(2).strip() if match.group(2) else None

    # Map string to ActionType
    action_map = {
        "observe": ActionType.OBSERVE,
        "mirror_traffic": ActionType.MIRROR_TRAFFIC,
        "redirect": ActionType.REDIRECT,
        "loud_contain": ActionType.LOUD_CONTAIN,
        "partial_covert": ActionType.PARTIAL_COVERT,
        "wait_and_track": ActionType.WAIT_AND_TRACK,
        "lock_artifact": ActionType.LOCK_ARTIFACT,
        "emergency_expel": ActionType.EMERGENCY_EXPEL,
    }

    action_type = action_map.get(action_str, ActionType.WAIT_AND_TRACK)
    target = target_str if target_str else None

    return SREAction(action_type=action_type, target=target)


def run_inference_episode(env: ShadowNetEnv, task_name: str,
                          generate_fn=None) -> dict:
    """
    Run one episode with LLM inference.

    Args:
        env: ShadowNetEnv instance
        task_name: task to run
        generate_fn: callable(system_prompt, user_prompt) -> str
                     If None, falls back to baseline agent.

    Returns: grade dict with reasoning log
    """
    obs = env.reset(task_name)
    reasoning_log = []

    # Create baseline agent once for the whole episode
    _baseline_agent = None
    if not generate_fn:
        from agent import BaselineAgent
        _baseline_agent = BaselineAgent()
        _baseline_agent.reset()

    # [START] logging — OpenEnv requirement
    print(f"[START] task={task_name} episode={env.state.episode_id}")

    step = 0
    while not obs.done:
        step += 1
        user_prompt = format_observation(obs)

        if generate_fn:
            # LLM generates reasoning + action
            response = generate_fn(SYSTEM_PROMPT, user_prompt)
            action = parse_action(response)
            reasoning_log.append({
                "step": step,
                "observation": user_prompt,
                "reasoning": response,
                "action": f"{action.action_type.value} {action.target or ''}".strip(),
            })
        else:
            # Fallback to baseline (persistent agent)
            action = _baseline_agent.act(obs)
            reasoning_log.append({
                "step": step,
                "action": f"{action.action_type.value} {action.target or ''}".strip(),
            })

        obs = env.step(action)

        # [STEP] logging — OpenEnv requirement
        reward_str = f" reward={obs.reward}" if obs.reward else ""
        print(f"[STEP] step={step} action={action.action_type.value} target={action.target or 'none'}{reward_str}")

    # [END] logging — OpenEnv requirement
    final_reward = obs.reward or compute_reward(env.state)
    print(f"[END] total_reward={final_reward} steps={step}")

    grade = grade_episode(env.state)
    grade["reasoning_log"] = reasoning_log

    return grade


def run_all_tasks(generate_fn=None):
    """Run inference on all 3 tasks. Prints [START][STEP][END] for each."""
    results = {}
    for task in ["shadow-easy", "shadow-medium", "shadow-hard"]:
        env = ShadowNetEnv()
        grade = run_inference_episode(env, task, generate_fn)
        results[task] = grade
        print(f"\n--- {task}: score={grade['score']} ---\n")

    return results


if __name__ == "__main__":
    # Run with baseline agent (no LLM) to generate required logs
    print("=" * 60)
    print("ShadowNet Inference — Baseline Agent")
    print("=" * 60)
    results = run_all_tasks(generate_fn=None)

    print("\n" + "=" * 60)
    print("Summary:")
    for task, grade in results.items():
        print(f"  {task}: {grade['score']}")
    print("=" * 60)
