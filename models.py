"""
ShadowNet — Typed Schemas
═══════════════════════════════════════════════════════════════
SREAction:       What the agent can do (8 action types)
SREObservation:  What the agent can SEE (no detection_risk!)
SREState:        Full internal state (hidden from agent)
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ─── ACTION SCHEMA ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """All 8 actions available to the Threat Containment Specialist."""
    OBSERVE = "observe"                    # Phase 1, 2 — reveal attacker activity on node
    MIRROR_TRAFFIC = "mirror_traffic"      # Phase 1, 2 — replicate node to honeypot, +fidelity
    REDIRECT = "redirect"                  # Phase 2    — move node to honeypot
    LOUD_CONTAIN = "loud_contain"          # Phase 2    — hard block, zero forensic
    PARTIAL_COVERT = "partial_covert"      # Phase 2    — emergency redirect, high risk spike
    WAIT_AND_TRACK = "wait_and_track"      # Phase 1, 2 — skip step, passive data, -detection
    LOCK_ARTIFACT = "lock_artifact"        # Phase 3    — secure forensic artifact
    EMERGENCY_EXPEL = "emergency_expel"    # Any        — end episode, score ~0.25


class SREAction(BaseModel):
    """Agent action with type and optional target."""
    action_type: ActionType
    target: Optional[str] = None  # node name or artifact id


# ─── OBSERVATION SCHEMA (WHAT THE AGENT SEES) ───────────────────────────────────

class SREObservation(BaseModel):
    """
    Observation returned to the agent after each step.

    CRITICAL: detection_risk is NEVER included here.
    The agent must INFER attacker suspicion from behavioral signals.
    This is the core mechanic — theory of mind, not state reading.
    """
    # Episode context
    step: int = 0
    phase: Literal["track", "contain", "evidence", "done"] = "track"
    max_steps: int = 20
    done: bool = False
    task_name: str = ""

    # Network topology — agent knows the network layout (this is public info)
    network_nodes: List[str] = Field(default_factory=list)    # all node names in scenario

    # Network visibility — what nodes the agent has observed
    visible_nodes: List[str] = Field(default_factory=list)
    anomalous_nodes: List[str] = Field(default_factory=list)  # nodes with attacker activity
    honeypot_nodes: List[str] = Field(default_factory=list)
    mirrored_nodes: List[str] = Field(default_factory=list)
    contained_nodes: List[str] = Field(default_factory=list)  # loud-contained

    # ─── BEHAVIORAL SIGNALS (proxy for hidden detection_risk) ───
    attacker_query_frequency: int = 2        # spikes when attacker suspicious
    honeypot_probe_count: int = 0            # increases as attacker tests consistency
    attacker_lateral_attempts: int = 0       # new movement = suspicion spike
    last_action_outcome: Literal[
        "smooth", "minor_anomaly", "flagged", "alerted"
    ] = "smooth"

    # Attacker behavior tier (derived from hidden risk, but observable)
    attacker_behavior: Literal[
        "passive", "cautious", "aggressive", "alerted"
    ] = "passive"

    # Attacker profile sampled for this episode (observable to agent)
    # Affects speed, noise level, pivot probability, detection sensitivity
    attacker_profile: Literal[
        "stealthy", "aggressive", "noisy", "adaptive"
    ] = "aggressive"

    # Honeypot fidelity — observable because agent's own mirror_traffic actions build it
    # Agent knows how many times it has mirrored; fidelity reflects that directly.
    # Hidden detection_risk is NOT included here.
    honeypot_fidelity: float = 0.85

    # Active alerts visible to agent
    active_alerts: List[str] = Field(default_factory=list)

    # Forensic artifacts available to lock (Phase 3)
    available_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    secured_artifacts: List[str] = Field(default_factory=list)

    # Reward (only populated when done=True)
    reward: Optional[float] = None

    # Valid actions this step
    valid_actions: List[str] = Field(default_factory=list)

    # Agent reasoning (populated by inference.py, not env)
    decision_log: Optional[str] = None


# ─── INTERNAL STATE (HIDDEN FROM AGENT) ──────────────────────────────────────────

class SREState(BaseModel):
    """
    Full internal environment state. Includes hidden variables.
    The agent NEVER receives this directly — only SREObservation.
    Exposed via /state endpoint for debugging only.
    """
    # Episode identity
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_name: str = ""
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    phase: Literal["track", "contain", "evidence", "done"] = "track"

    # ─── HIDDEN: Detection Risk Engine ───
    # Agent CANNOT see these values. Must infer from behavioral signals.
    detection_risk: float = 0.0            # 0.0 to 1.0, hidden
    peak_detection_risk: float = 0.0       # tracks worst moment
    attacker_alerted: bool = False         # True if risk >= 0.85
    logs_deleted: bool = False
    forensic_value_cap: float = 1.0        # drops to 0.1 if attacker alerted

    # ─── Honeypot State ───
    honeypot_fidelity: float = 0.85        # starts at 0.85, mirror_traffic raises it
    honeypot_probe_count: int = 0
    last_action_outcome: str = "smooth"

    # ─── Network State ───
    all_nodes: List[str] = Field(default_factory=list)
    probed_nodes: List[str] = Field(default_factory=list)       # attacker visited
    observed_nodes: List[str] = Field(default_factory=list)     # agent observed
    mirroring_nodes: List[str] = Field(default_factory=list)    # being mirrored
    honeypot_nodes: List[str] = Field(default_factory=list)     # redirected to honeypot
    compromised_nodes: List[str] = Field(default_factory=list)  # loud-contained
    locked_artifacts: List[str] = Field(default_factory=list)
    attacker_current_node: str = ""

    # ─── Reward Components ───
    real_data_accessed: int = 0
    total_real_assets: int = 0
    artifacts_secured: int = 0
    total_artifacts: int = 0

    # ─── SIEM Log ───
    siem_log: List[Dict[str, Any]] = Field(default_factory=list)
    active_alerts: List[str] = Field(default_factory=list)

    # ─── Tracking ───
    actions_this_step: int = 0
    mirrored_before_redirect: Dict[str, bool] = Field(default_factory=dict)
    emergency_expel_used: bool = False


# ─── SCENARIO DATA TYPES ────────────────────────────────────────────────────────

class NodeDef(BaseModel):
    """A network node in a scenario."""
    name: str
    sensitivity: int = 5                    # 0-10, how critical this asset is
    x: int = 0                              # SVG canvas x position
    y: int = 0                              # SVG canvas y position
    artifacts: List[str] = Field(default_factory=list)  # forensic artifacts available
    artifact_decay: Dict[str, int] = Field(default_factory=dict)  # artifact -> steps to decay


class AttackerStep(BaseModel):
    """One step in the attacker's scripted path."""
    step: int                               # which env step this fires on
    action: Literal["probe", "escalate", "pivot", "exfiltrate", "lateral", "persist"]
    target: str                             # node name


class EdgeDef(BaseModel):
    """Network edge connecting two nodes."""
    source: str  # using 'source' instead of 'from' (reserved keyword)
    target: str  # using 'target' instead of 'to'


class Scenario(BaseModel):
    """Complete incident scenario definition."""
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    nodes: List[NodeDef]
    edges: List[EdgeDef]
    attacker_path: List[AttackerStep]
    honeypot_fidelity_start: float = 0.85
    critical_asset: Optional[str] = None    # the ONE node that must not be breached
    attack_source: str = ""                 # real-world breach this is based on
