"""
ShadowNet — 6-Signal Reward Calculator
═══════════════════════════════════════════════════════════════
Signal          Weight  What It Measures
─────────────── ──────  ──────────────────────────────────────
asset_safety      0.25  Did real data stay protected?
forensic_value    0.25  How much evidence was captured?
stealth_score     0.20  How close did detection risk get to threshold?
honeypot_quality  0.15  Was the honeypot trap convincing & actively used?
phase_completion  0.10  Did the agent complete all 3 phases?
efficiency        0.05  How quickly was the mission completed?

ALL scores clamped to (0.01, 0.99) — OpenEnv requirement.
A reward of exactly 0.0 or 1.0 FAILS the automated checker.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Dict
from models import SREState


def compute_reward(state: SREState) -> float:
    """
    Compute final episode reward from 6 signals.

    Returns: float in (0.01, 0.99) — NEVER 0.0 or 1.0.
    """
    c = compute_reward_components(state)

    raw = (c["asset_safety"]      * 0.25
         + c["forensic_value"]    * 0.25
         + c["stealth_score"]     * 0.20
         + c["honeypot_quality"]  * 0.15
         + c["phase_completion"]  * 0.10
         + c["efficiency"]        * 0.05)

    return _clamp(raw)


def compute_reward_components(state: SREState) -> Dict[str, float]:
    """
    Compute individual reward components (all normalized to 0–1).
    Useful for debugging, comparison tables, and the /grader endpoint.
    """

    # ── 1. ASSET SAFETY (25%) ─────────────────────────────────────────────
    # Core objective: prevent attacker from reading real sensitive data.
    # Penalty per real asset accessed scales non-linearly:
    # first breach is harsh, subsequent breaches still matter but less so.
    total_assets = max(state.total_real_assets, 1)
    breach_ratio = state.real_data_accessed / total_assets
    # Non-linear: sqrt makes early breaches hurt more
    asset_safety = max(0.0, 1.0 - (breach_ratio ** 0.75))
    # Emergency expel = attacker gone but we learned nothing = partial credit
    if state.emergency_expel_used:
        asset_safety = min(asset_safety, 0.6)

    # ── 2. FORENSIC VALUE (25%) ───────────────────────────────────────────
    # Measures intelligence gathered. Capped by forensic_value_cap
    # (set to 0.1 if attacker went dark, 0.0 if emergency expel).
    total_artifacts = max(state.total_artifacts, 1)
    forensic_raw = state.artifacts_secured / total_artifacts
    forensic_value = min(forensic_raw, state.forensic_value_cap)
    if state.emergency_expel_used:
        forensic_value = 0.0

    # ── 3. STEALTH SCORE (20%) ────────────────────────────────────────────
    # Measures how well the agent avoided triggering the attacker.
    # Non-linear: keeping risk below 0.5 is rewarded heavily,
    # approaching 0.85 (threshold) is penalized exponentially.
    peak = state.peak_detection_risk
    if peak < 0.5:
        stealth_score = 1.0 - (peak * 0.4)         # mild penalty below 0.5
    elif peak < 0.85:
        stealth_score = 0.8 - ((peak - 0.5) * 1.71) # steeper penalty 0.5→0.85
    else:
        stealth_score = 0.1                          # attacker alerted = near-zero

    # ── 4. HONEYPOT QUALITY (15%) ─────────────────────────────────────────
    # Rewards agents that properly built and used the honeypot trap:
    # mirrored traffic first, redirected into it, maintained fidelity.
    honeypot_nodes = len(state.honeypot_nodes)
    mirroring_nodes = len(state.mirroring_nodes)
    fidelity = getattr(state, "honeypot_fidelity", 0.85)

    # Base: did any honeypots get created?
    if honeypot_nodes == 0:
        honeypot_quality = 0.0
    else:
        # Bonus for mirroring before redirecting (proper technique)
        mirror_ratio = mirroring_nodes / max(honeypot_nodes, 1)
        # Bonus for maintaining high fidelity (attacker didn't notice inconsistencies)
        honeypot_quality = (0.5 * min(fidelity, 1.0)
                          + 0.3 * min(mirror_ratio, 1.0)
                          + 0.2 * min(honeypot_nodes / max(state.total_real_assets, 1), 1.0))

    # ── 5. PHASE COMPLETION (10%) ─────────────────────────────────────────
    # Rewards completing the full 3-phase arc (track → contain → evidence).
    # An agent that emergency-expels at step 1 should score much lower
    # than one that completes all 3 phases even with mediocre sub-scores.
    phase_scores = {
        "track":    0.2,   # only completed tracking
        "contain":  0.5,   # reached containment phase
        "evidence": 0.85,  # reached evidence phase
        "done":     1.0,   # completed full episode
    }
    phase_completion = phase_scores.get(state.phase, 0.2)
    if state.emergency_expel_used:
        phase_completion = min(phase_completion, 0.3)

    # ── 6. EFFICIENCY (5%) ────────────────────────────────────────────────
    # Partial credit always given — using all steps is not a failure.
    # Formula: agent gets 50% of this signal even at max steps.
    max_steps = max(state.max_steps, 1)
    efficiency = max(0.0, 1.0 - (state.step_count / max_steps * 0.5))

    return {
        "asset_safety":     round(max(0.0, min(1.0, asset_safety)),     4),
        "forensic_value":   round(max(0.0, min(1.0, forensic_value)),   4),
        "stealth_score":    round(max(0.0, min(1.0, stealth_score)),    4),
        "honeypot_quality": round(max(0.0, min(1.0, honeypot_quality)), 4),
        "phase_completion": round(max(0.0, min(1.0, phase_completion)), 4),
        "efficiency":       round(max(0.0, min(1.0, efficiency)),       4),
    }


def _clamp(raw_score: float) -> float:
    """
    Clamp score to (0.01, 0.99).

    CRITICAL: OpenEnv automated checker rejects exactly 0.0 or 1.0.
    This function ensures we NEVER return those values regardless of inputs.
    """
    return round(min(0.99, max(0.01, raw_score)), 3)


def grade_episode(state: SREState) -> Dict:
    """
    Full grading output for /grader endpoint.
    Returns score, all 6 component breakdowns, weights, and metadata.
    """
    components = compute_reward_components(state)
    final = compute_reward(state)

    # Classify outcome for the judging dashboard
    if state.emergency_expel_used:
        outcome = "emergency_expel"
    elif state.attacker_alerted:
        outcome = "attacker_alerted"
    elif state.phase == "done":
        outcome = "clean_success"
    else:
        outcome = "timeout"

    return {
        "score": final,
        "outcome": outcome,
        "components": components,
        "weights": {
            "asset_safety":     0.25,
            "forensic_value":   0.25,
            "stealth_score":    0.20,
            "honeypot_quality": 0.15,
            "phase_completion": 0.10,
            "efficiency":       0.05,
        },
        "metadata": {
            "steps_taken":         state.step_count,
            "max_steps":           state.max_steps,
            "phase_reached":       state.phase,
            "attacker_alerted":    state.attacker_alerted,
            "emergency_expel":     state.emergency_expel_used,
            "artifacts_secured":   state.artifacts_secured,
            "total_artifacts":     state.total_artifacts,
            "honeypot_nodes":      len(state.honeypot_nodes),
            "mirroring_nodes":     len(state.mirroring_nodes),
            "honeypot_fidelity":   round(getattr(state, "honeypot_fidelity", 0.85), 3),
            "real_data_accessed":  state.real_data_accessed,
            "total_real_assets":   state.total_real_assets,
            "peak_detection_risk": round(state.peak_detection_risk, 3),
        },
    }
