"""
Prompt builders for ShadowNet training and evaluation.
"""

from __future__ import annotations

from models import SREAction, SREObservation


SYSTEM_PROMPT = """You are a Threat Containment Specialist operating inside ShadowNet.
Your mission is to contain an active network intruder without alerting them.

You cannot see the attacker's hidden suspicion level. You must infer it from:
- attacker_query_frequency
- honeypot_probe_count
- attacker_lateral_attempts
- last_action_outcome
- attacker_behavior

Output exactly:
ACTION: <action_name>
TARGET: <target_or_none>

Optional:
REASON: <brief justification>

Rules:
- prefer mirror_traffic before redirect
- if suspicion signals are high, wait_and_track is often safer
- in evidence phase, lock artifacts with the lowest remaining TTL first
- only use emergency_expel as a last resort
"""


def build_observation_prompt(obs: SREObservation) -> str:
    """Render an observation as a stable training/eval prompt."""
    artifacts = (
        ", ".join(f"{a['id']}({a.get('steps_remaining', '?')})" for a in obs.available_artifacts)
        if obs.available_artifacts
        else "none"
    )
    alerts = ", ".join(obs.active_alerts) if obs.active_alerts else "none"

    return "\n".join(
        [
            f"PHASE: {obs.phase}",
            f"STEP: {obs.step} / {obs.max_steps}",
            f"TASK: {obs.task_name}",
            f"VALID_ACTIONS: {', '.join(obs.valid_actions) if obs.valid_actions else 'none'}",
            f"NETWORK_NODES: {', '.join(obs.network_nodes) if obs.network_nodes else 'none'}",
            f"VISIBLE_NODES: {', '.join(obs.visible_nodes) if obs.visible_nodes else 'none'}",
            f"ANOMALOUS_NODES: {', '.join(obs.anomalous_nodes) if obs.anomalous_nodes else 'none'}",
            f"MIRRORED_NODES: {', '.join(obs.mirrored_nodes) if obs.mirrored_nodes else 'none'}",
            f"HONEYPOT_NODES: {', '.join(obs.honeypot_nodes) if obs.honeypot_nodes else 'none'}",
            f"CONTAINED_NODES: {', '.join(obs.contained_nodes) if obs.contained_nodes else 'none'}",
            f"ATTACKER_BEHAVIOR: {obs.attacker_behavior}",
            f"ATTACKER_PROFILE: {obs.attacker_profile}",
            f"QUERY_FREQUENCY: {obs.attacker_query_frequency}",
            f"PROBE_COUNT: {obs.honeypot_probe_count}",
            f"LATERAL_ATTEMPTS: {obs.attacker_lateral_attempts}",
            f"HONEYPOT_FIDELITY: {obs.honeypot_fidelity:.2f}",
            f"LAST_ACTION_OUTCOME: {obs.last_action_outcome}",
            f"ACTIVE_ALERTS: {alerts}",
            f"AVAILABLE_ARTIFACTS: {artifacts}",
            f"SECURED_ARTIFACTS: {', '.join(obs.secured_artifacts) if obs.secured_artifacts else 'none'}",
        ]
    )


def build_chat_messages(obs: SREObservation) -> list[dict[str, str]]:
    """Return chat messages suitable for chat-template based generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_observation_prompt(obs)},
    ]


def format_action_block(action: SREAction, reason: str | None = None) -> str:
    """Format an action in the canonical structured output style."""
    lines = [
        f"ACTION: {action.action_type.value}",
        f"TARGET: {action.target if action.target is not None else 'none'}",
    ]
    if reason:
        lines.append(f"REASON: {reason}")
    return "\n".join(lines)
