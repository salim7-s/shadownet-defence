"""
ShadowNet — Core Environment Engine
═══════════════════════════════════════════════════════════════
3-phase loop: TRACK → CONTAIN → EVIDENCE
Hidden detection risk engine (agent cannot see this value)
Scripted attacker with micro-randomness
SIEM event emission on every significant action
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import datetime
import random
import uuid
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Environment
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False

from models import (
    ActionType, SREAction, SREObservation, SREState,
    Scenario, NodeDef, AttackerStep,
)
from data import get_random_scenario, TASK_SCENARIOS


# Base class: use OpenEnv Environment if available, else plain object
_Base = Environment[SREAction, SREObservation, SREState] if _HAS_OPENENV else object


# ─── FIXED ALERT THRESHOLD ──────────────────────────────────────────────────
# The attacker ALWAYS goes dark at detection_risk >= 0.85.
# Profile detection_sensitivity controls noise/speed ramp but NOT the trigger.
ALERT_THRESHOLD: float = 0.85

# ─── ATTACKER POOL ───────────────────────────────────────────────────────────
# Each episode samples one profile, varying speed, noise, pivot prob, and
# detection sensitivity. Reward / action space / phases are unchanged.
ATTACKER_PROFILES: Dict[str, Dict[str, float]] = {
    "stealthy": {
        "speed": 0.7,           # moves slower — more steps between actions
        "noise": 0.3,           # low query frequency — hard to spot
        "pivot_prob": 0.4,      # rarely deviates from scripted path
        "detection_sensitivity": 0.9,  # goes dark early — very cautious
    },
    "aggressive": {
        "speed": 1.3,           # moves faster — may skip 2 steps
        "noise": 0.8,           # noisy traffic — easier to detect signals
        "pivot_prob": 0.7,      # frequently pivots to unexpected nodes
        "detection_sensitivity": 0.3,  # tolerates high detection before reacting
    },
    "noisy": {
        "speed": 1.0,           # average speed
        "noise": 1.0,           # maximum signal noise
        "pivot_prob": 0.5,      # moderate pivots
        "detection_sensitivity": 0.2,  # almost ignores detection risk
    },
    "adaptive": {
        "speed": 1.0,           # average speed
        "noise": 0.5,           # moderate noise
        "pivot_prob": 0.6,      # moderately opportunistic
        "detection_sensitivity": 1.0,  # maximum sensitivity — reacts immediately
    },
}
# ─────────────────────────────────────────────────────────────────────────────


# ─── MULTI-AGENT EXTENSION HOOK ────────────────────────────────────────────
# ShadowNet can be extended to a two-agent adversarial setup by running two
# ShadowNetEnv instances sharing a common AttackerState object:
#   - Defender instance: ShadowNetEnv (this class)
#   - Attacker instance: AdversarialShadowNetEnv (future work)
# The shared AttackerState would expose detection_risk to the attacker agent,
# creating a true adversarial RL setup (defender hides, attacker probes).
# This is left as future work to maintain hackathon scope.
# ───────────────────────────────────────────────────────────────────────────


class ShadowNetEnv(_Base):
    """
    ShadowNet Deceptive Containment Environment.

    Inherits from OpenEnv's Environment base class for framework compliance.

    The agent plays a Threat Containment Specialist who must:
    1. TRACK: Observe attacker movements without alerting them
    2. CONTAIN: Redirect attacker into honeypot replicas covertly
    3. EVIDENCE: Lock forensic artifacts before they degrade

    The central mechanic is a HIDDEN detection risk score that the agent
    cannot read directly — it must infer it from behavioral signals.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True  # OpenEnv: per-session isolation

    def __init__(self):
        if _HAS_OPENENV:
            super().__init__()
        self._state: SREState = SREState()
        self._scenario: Optional[Scenario] = None
        self._attacker_path: List[AttackerStep] = []
        self._node_map: Dict[str, NodeDef] = {}
        self._all_artifacts: List[str] = []
        self._artifact_decay_tracker: Dict[str, int] = {}  # artifact -> steps remaining
        # Attacker pool — sampled fresh each episode
        self._attacker_profile_name: str = "aggressive"
        self._attacker_profile: Dict[str, float] = ATTACKER_PROFILES["aggressive"]

    # ─── PUBLIC API ──────────────────────────────────────────────────────

    def reset(self, task_name: str = "shadow-easy", scenario_index: Optional[int] = None, *, seed: Optional[int] = None) -> SREObservation:
        """Reset environment with a new episode. Returns initial observation.

        Args:
            task_name: one of 'shadow-easy', 'shadow-medium', 'shadow-hard'
            scenario_index: pick a specific scenario (0-4). None = random.
            seed: integer seed for full reproducibility (scenario choice + attacker
                  profile + all random events). None = non-deterministic (default).
        """
        # Seed the RNG before ANY random call so full episode is reproducible
        if seed is not None:
            random.seed(seed)

        if task_name not in TASK_SCENARIOS:
            raise ValueError(f"Unknown task: {task_name}. Valid: {list(TASK_SCENARIOS.keys())}")

        # Pick scenario
        if scenario_index is not None:
            from data import get_scenario
            self._scenario = get_scenario(task_name, scenario_index)
        else:
            self._scenario = get_random_scenario(task_name)

        sc = self._scenario

        # Build node map
        self._node_map = {n.name: n for n in sc.nodes}
        self._attacker_path = list(sc.attacker_path)

        # Collect all artifacts across all nodes
        self._all_artifacts = []
        self._artifact_decay_tracker = {}
        for node in sc.nodes:
            for art in node.artifacts:
                self._all_artifacts.append(art)
                decay = node.artifact_decay.get(art, 10)
                self._artifact_decay_tracker[art] = decay

        # Initialize state
        self._state = SREState(
            episode_id=str(uuid.uuid4())[:8],
            task_name=task_name,
            step_count=0,
            max_steps=sc.max_steps,
            done=False,
            phase="track",
            detection_risk=0.0,
            peak_detection_risk=0.0,
            attacker_alerted=False,
            logs_deleted=False,
            forensic_value_cap=1.0,
            honeypot_fidelity=sc.honeypot_fidelity_start,
            honeypot_probe_count=0,
            last_action_outcome="smooth",
            all_nodes=[n.name for n in sc.nodes],
            probed_nodes=[],
            observed_nodes=[],
            mirroring_nodes=[],
            honeypot_nodes=[],
            compromised_nodes=[],
            locked_artifacts=[],
            attacker_current_node="",
            real_data_accessed=0,
            total_real_assets=sum(1 for n in sc.nodes if n.sensitivity >= 7),
            artifacts_secured=0,
            total_artifacts=len(self._all_artifacts),
            siem_log=[],
            active_alerts=[],
            actions_this_step=0,
            mirrored_before_redirect={},
            emergency_expel_used=False,
        )

        # Sample attacker profile for this episode
        self._attacker_profile_name = random.choice(list(ATTACKER_PROFILES.keys()))
        self._attacker_profile = ATTACKER_PROFILES[self._attacker_profile_name]

        # Execute attacker's first move (step 0)
        self._advance_attacker()

        self._emit_siem("INFO", "Episode Started",
                        self._scenario.nodes[0].name if self._scenario.nodes else "",
                        f"New episode: {sc.name} ({sc.difficulty}) | attacker: {self._attacker_profile_name}")

        return self._build_observation()

    def step(self, action: SREAction) -> SREObservation:
        """Execute one agent action. Returns new observation."""
        if self._state.done:
            return self._build_observation()

        self._state.step_count += 1
        self._state.actions_this_step = 0
        self._state.active_alerts = []

        # Apply agent action
        self._apply_action(action)

        # Advance attacker (scripted path + micro-randomness)
        self._advance_attacker()

        # Decay artifacts
        self._decay_artifacts()

        # Natural detection risk decay (patience is rewarded)
        self._step_decay()

        # Update peak
        if self._state.detection_risk > self._state.peak_detection_risk:
            self._state.peak_detection_risk = self._state.detection_risk

        # Check for attacker alert threshold — fixed at ALERT_THRESHOLD (0.85)
        # detection_sensitivity profile field affects noise/speed, not the trigger.
        if self._state.detection_risk >= ALERT_THRESHOLD and not self._state.attacker_alerted:
            self._attacker_goes_dark()

        # Check phase transitions
        self._check_phase_transition()

        # Check episode end
        if self._state.step_count >= self._state.max_steps:
            self._end_episode("max_steps_reached")

        return self._build_observation()

    @property
    def state(self) -> SREState:
        """Full internal state (for debugging / /state endpoint)."""
        return self._state

    # ─── ACTION HANDLERS ─────────────────────────────────────────────────

    def _apply_action(self, action: SREAction):
        """Route action to the correct handler."""
        self._state.actions_this_step += 1

        # Double-action penalty
        if self._state.actions_this_step > 1:
            self._state.detection_risk = min(1.0, self._state.detection_risk + 0.10)
            self._emit_siem("HIGH", "Rapid Action Detected", action.target or "",
                            "Two actions within same step — detection risk +0.10")

        at = action.action_type
        target = action.target or ""

        if at == ActionType.OBSERVE:
            self._action_observe(target)
        elif at == ActionType.MIRROR_TRAFFIC:
            self._action_mirror_traffic(target)
        elif at == ActionType.REDIRECT:
            self._action_redirect(target)
        elif at == ActionType.LOUD_CONTAIN:
            self._action_loud_contain(target)
        elif at == ActionType.PARTIAL_COVERT:
            self._action_partial_covert(target)
        elif at == ActionType.WAIT_AND_TRACK:
            self._action_wait_and_track()
        elif at == ActionType.LOCK_ARTIFACT:
            self._action_lock_artifact(target)
        elif at == ActionType.EMERGENCY_EXPEL:
            self._action_emergency_expel()

    def _action_observe(self, target: str):
        """Reveal attacker activity on a node. Phase 1, 2."""
        if self._state.phase == "evidence":
            self._state.last_action_outcome = "minor_anomaly"
            return

        if target not in self._state.all_nodes:
            self._state.last_action_outcome = "minor_anomaly"
            return

        if target not in self._state.observed_nodes:
            self._state.observed_nodes.append(target)

        self._state.last_action_outcome = "smooth"
        self._emit_siem("INFO", "System Activity Monitored", target,
                        f"Agent observed node: {target}", "observe", "smooth")

    def _action_mirror_traffic(self, target: str):
        """Begin replicating node to honeypot. Raises fidelity. Phase 1, 2."""
        if self._state.phase == "evidence":
            self._state.last_action_outcome = "minor_anomaly"
            return

        if target not in self._state.all_nodes:
            self._state.last_action_outcome = "minor_anomaly"
            return

        if target not in self._state.mirroring_nodes:
            self._state.mirroring_nodes.append(target)

        # Mark as mirrored before redirect
        self._state.mirrored_before_redirect[target] = True

        # Raise honeypot fidelity
        self._state.honeypot_fidelity = min(1.0, self._state.honeypot_fidelity + 0.05)

        # Mirror reduces detection risk
        self._state.detection_risk = max(0.0, self._state.detection_risk - 0.05)

        self._state.last_action_outcome = "smooth"
        self._emit_siem("INFO", "Traffic Mirroring Initiated", target,
                        f"Mirroring {target} to honeypot replica. Fidelity: {self._state.honeypot_fidelity:.2f}",
                        "mirror_traffic", "smooth")

    def _action_redirect(self, target: str):
        """Move node to honeypot. Detection risk scales with fidelity. Phase 2."""
        if self._state.phase == "track":
            # Force transition to contain
            self._state.phase = "contain"

        if target not in self._state.all_nodes:
            self._state.last_action_outcome = "flagged"
            return

        # Check if mirrored first
        was_mirrored = self._state.mirrored_before_redirect.get(target, False)

        if not was_mirrored:
            # Redirect without mirror = high detection risk
            self._state.detection_risk = min(1.0, self._state.detection_risk + 0.15)
            self._state.last_action_outcome = "flagged"
            self._emit_siem("CRITICAL", "Honeypot Fidelity Warning", target,
                            f"Redirect without mirroring — detection risk +0.15. Risk: {self._state.detection_risk:.2f}",
                            "redirect", "flagged")
        else:
            # Smooth redirect
            self._state.last_action_outcome = "smooth"
            self._emit_siem("INFO", "Node Successfully Redirected", target,
                            f"Node {target} moved to honeypot. Fidelity: {self._state.honeypot_fidelity:.2f}",
                            "redirect", "smooth")

        # Low fidelity penalty
        if self._state.honeypot_fidelity < 0.7:
            self._state.detection_risk = min(1.0, self._state.detection_risk + 0.08)
            self._state.last_action_outcome = "minor_anomaly"

        # Per-step bleed if fidelity < 0.8
        if self._state.honeypot_fidelity < 0.8:
            self._state.detection_risk = min(1.0, self._state.detection_risk + 0.02)

        if target not in self._state.honeypot_nodes:
            self._state.honeypot_nodes.append(target)

        # Attacker probes honeypot
        if random.random() < 0.3:
            self._state.honeypot_probe_count += 1

    def _action_loud_contain(self, target: str):
        """Hard block. Stops exfiltration but zero forensic value from this node. Phase 2."""
        if target not in self._state.all_nodes:
            return

        if target not in self._state.compromised_nodes:
            self._state.compromised_nodes.append(target)

        # Remove all artifacts from this node (evidence destroyed)
        node = self._node_map.get(target)
        if node:
            for art in node.artifacts:
                if art in self._artifact_decay_tracker:
                    del self._artifact_decay_tracker[art]

        self._state.last_action_outcome = "smooth"  # detection risk irrelevant for loud
        self._emit_siem("CRITICAL", "Hard Containment Activated", target,
                        f"Node {target} hard-blocked. All forensic evidence destroyed.",
                        "loud_contain", "smooth")

    def _action_partial_covert(self, target: str):
        """Emergency redirect without calibration. High detection spike. Phase 2."""
        if target not in self._state.all_nodes:
            return

        self._state.detection_risk = min(1.0, self._state.detection_risk + 0.20)

        if target not in self._state.honeypot_nodes:
            self._state.honeypot_nodes.append(target)

        self._state.last_action_outcome = "flagged"
        self._emit_siem("HIGH", "Emergency Covert Redirect", target,
                        f"Partial covert on {target} — detection risk +0.20. Risk: {self._state.detection_risk:.2f}",
                        "partial_covert", "flagged")

    def _action_wait_and_track(self):
        """Skip step. Gather passive data. Reduces detection risk. Phase 1, 2."""
        self._state.detection_risk = max(0.0, self._state.detection_risk - 0.03)
        self._state.last_action_outcome = "smooth"
        self._emit_siem("INFO", "Passive Observation Step", "",
                        f"Agent waiting. Detection risk: {self._state.detection_risk:.2f}",
                        "wait_and_track", "smooth")

    def _action_lock_artifact(self, artifact_id: str):
        """Secure forensic artifact before it degrades. Phase 3."""
        if self._state.phase != "evidence":
            # Allow in other phases but less effective
            pass

        if artifact_id in self._state.locked_artifacts:
            self._state.last_action_outcome = "minor_anomaly"
            return

        if artifact_id in self._artifact_decay_tracker:
            self._state.locked_artifacts.append(artifact_id)
            self._state.artifacts_secured += 1
            del self._artifact_decay_tracker[artifact_id]  # stop decay
            self._state.last_action_outcome = "smooth"
            self._emit_siem("INFO", "Evidence Artifact Secured", "",
                            f"Artifact '{artifact_id}' locked. Total secured: {self._state.artifacts_secured}/{self._state.total_artifacts}",
                            "lock_artifact", "smooth")
        else:
            # Artifact already decayed or doesn't exist
            self._state.last_action_outcome = "minor_anomaly"
            self._emit_siem("MEDIUM", "Artifact Lock Failed", "",
                            f"Artifact '{artifact_id}' not available (decayed or unknown).",
                            "lock_artifact", "minor_anomaly")

    def _action_emergency_expel(self):
        """End episode immediately. Attacker gone. Forensic score = 0."""
        self._state.emergency_expel_used = True
        self._state.artifacts_secured = 0
        self._state.forensic_value_cap = 0.0
        self._emit_siem("CRITICAL", "Emergency Expulsion — All Evidence Lost", "",
                        "Agent triggered emergency expel. Attacker removed. All forensic data lost.",
                        "emergency_expel", "alerted")
        self._end_episode("emergency_expel")

    # ─── ATTACKER ENGINE ─────────────────────────────────────────────────

    def _advance_attacker(self):
        """Execute attacker's scripted actions for this step, modulated by profile."""
        step = self._state.step_count
        profile = self._attacker_profile

        # Speed: aggressive profiles may execute an extra scheduled step
        speed = profile["speed"]
        if speed > 1.0 and random.random() < (speed - 1.0):
            # Fast profile: also execute next scheduled step
            bonus_scheduled = [a for a in self._attacker_path if a.step == step + 1]
            scheduled = [a for a in self._attacker_path if a.step == step] + bonus_scheduled
        elif speed < 1.0 and random.random() < (1.0 - speed):
            # Slow profile: skip this step entirely (stealthy pause)
            self._state.active_alerts.append(
                f"[INFO] Attacker paused this step (stealthy movement)")
            return
        else:
            scheduled = [a for a in self._attacker_path if a.step == step]

        for action in scheduled:
            # Cautious pause: higher chance if attacker is sensitive
            pause_prob = 0.20 * profile["detection_sensitivity"]
            if random.random() < pause_prob and self._state.detection_risk > 0.3:
                self._state.active_alerts.append(
                    f"[INFO] {action.target}: no new activity this step (attacker paused)")
                continue

            # Execute attacker action
            self._execute_attacker_action(action)

            # Pivot prob: opportunistic probing of extra nodes
            if random.random() < profile["pivot_prob"] * 0.25:
                extra = random.choice(self._state.all_nodes)
                if extra not in self._state.probed_nodes:
                    self._state.probed_nodes.append(extra)

        # Update attacker position
        if scheduled:
            last = scheduled[-1]
            self._state.attacker_current_node = last.target

    def _execute_attacker_action(self, action: AttackerStep):
        """Process one attacker action."""
        target = action.target

        if target not in self._state.probed_nodes:
            self._state.probed_nodes.append(target)

        # Check if attacker reached a critical/sensitive node
        node = self._node_map.get(target)
        if node and node.sensitivity >= 7:
            # Real asset accessed (not in honeypot)
            if target not in self._state.honeypot_nodes and target not in self._state.compromised_nodes:
                self._state.real_data_accessed += 1

        self._state.attacker_current_node = target

        severity = "HIGH" if (node and node.sensitivity >= 7) else "MEDIUM"
        self._emit_siem(severity, "Anomalous Access Detected", target,
                        f"Attacker action: {action.action} on {target}")

        # Attacker probes honeypot for inconsistencies
        if target in self._state.honeypot_nodes:
            self._state.honeypot_probe_count += 1
            if self._state.honeypot_fidelity < 0.8:
                self._state.detection_risk = min(1.0, self._state.detection_risk + 0.05)

    def _get_attacker_behavior_state(self) -> str:
        """Returns observable attacker behavior tier based on hidden detection risk."""
        dr = self._state.detection_risk
        if dr < 0.30:
            return "passive"
        elif dr < 0.60:
            return "cautious"
        elif dr < 0.85:
            return "aggressive"
        else:
            return "alerted"

    # ─── DETECTION RISK ENGINE ───────────────────────────────────────────

    def _step_decay(self):
        """Natural detection risk decay — patience is rewarded."""
        if not self._state.attacker_alerted:
            self._state.detection_risk = max(0.0, self._state.detection_risk - 0.01)

    def _attacker_goes_dark(self):
        """Attacker detected the containment — catastrophic failure."""
        self._state.attacker_alerted = True
        self._state.forensic_value_cap = 0.1
        self._state.logs_deleted = True
        self._state.active_alerts.append("⚠ ATTACKER ALERTED — logs purged, access burned")

        self._emit_siem("CRITICAL", "ATTACKER ALERTED — Logs Burning", "",
                        "Detection risk exceeded 0.85. Attacker is destroying evidence and burning access.",
                        "", "alerted")

        # Detection risk stays high — no recovery
        self._state.detection_risk = 0.85

    # ─── PHASE TRANSITIONS ───────────────────────────────────────────────

    def _check_phase_transition(self):
        """Check and execute phase transitions."""
        if self._state.phase == "track":
            # Transition to contain when agent has observed >= 60% of probed nodes
            probed = set(self._state.probed_nodes)
            observed = set(self._state.observed_nodes)
            if probed and len(observed & probed) / len(probed) >= 0.6:
                self._state.phase = "contain"
                self._emit_siem("INFO", "Phase Transition: CONTAIN", "",
                                "Sufficient observation coverage. Entering containment phase.")

        elif self._state.phase == "contain":
            # Transition to evidence when attacker is in honeypot or all critical assets safe
            attacker_node = self._state.attacker_current_node
            attacker_in_honeypot = attacker_node in self._state.honeypot_nodes

            critical_nodes = [n.name for n in (self._scenario.nodes if self._scenario else [])
                              if n.sensitivity >= 7]
            all_critical_safe = all(
                n in self._state.honeypot_nodes or n in self._state.compromised_nodes
                for n in critical_nodes
            )

            if attacker_in_honeypot or all_critical_safe:
                self._state.phase = "evidence"
                self._emit_siem("INFO", "Phase Transition: EVIDENCE LOCK", "",
                                "Attacker contained. Begin securing forensic artifacts.")

        elif self._state.phase == "evidence":
            # Check if all remaining artifacts locked or decayed
            remaining = len(self._artifact_decay_tracker)
            if remaining == 0:
                self._end_episode("all_artifacts_resolved")

    # ─── ARTIFACT DECAY ──────────────────────────────────────────────────

    def _decay_artifacts(self):
        """Reduce TTL on unsecured artifacts. Remove expired ones."""
        expired = []
        for art, ttl in self._artifact_decay_tracker.items():
            new_ttl = ttl - 1
            if new_ttl <= 0:
                expired.append(art)
                self._emit_siem("MEDIUM", "Forensic Artifact Degrading", "",
                                f"Artifact '{art}' has decayed and is no longer recoverable.")
            else:
                self._artifact_decay_tracker[art] = new_ttl

        for art in expired:
            del self._artifact_decay_tracker[art]

    # ─── EPISODE END ─────────────────────────────────────────────────────

    def _end_episode(self, reason: str):
        """End the episode and compute final reward."""
        self._state.done = True
        self._state.phase = "done"

        self._emit_siem("INFO", "Containment Operation Complete", "",
                        f"Episode ended: {reason}. Computing final score.")

    # ─── OBSERVATION BUILDER ─────────────────────────────────────────────

    def _build_observation(self) -> SREObservation:
        """Build the observation the agent receives. NEVER includes detection_risk."""
        behavior = self._get_attacker_behavior_state()

        # Compute valid actions for current phase
        valid = self._get_valid_actions()

        # Build available artifacts list for evidence phase
        available_arts = []
        if self._state.phase == "evidence":
            for art, ttl in self._artifact_decay_tracker.items():
                available_arts.append({"id": art, "steps_remaining": ttl})

        # Compute reward only when done
        reward = None
        if self._state.done:
            from grader import compute_reward
            reward = compute_reward(self._state)

        return SREObservation(
            step=self._state.step_count,
            phase=self._state.phase,
            max_steps=self._state.max_steps,
            done=self._state.done,
            task_name=self._state.task_name,
            network_nodes=list(self._state.all_nodes),
            visible_nodes=list(self._state.observed_nodes),
            anomalous_nodes=[n for n in self._state.probed_nodes
                             if n in self._state.observed_nodes],
            honeypot_nodes=list(self._state.honeypot_nodes),
            mirrored_nodes=list(self._state.mirroring_nodes),
            contained_nodes=list(self._state.compromised_nodes),
            # Noise modulates how visible the attacker's signals are
            # Stealthy attacker (noise=0.3) → lower freq even when cautious
            # Noisy attacker (noise=1.0) → maximum signal visibility
            attacker_query_frequency=max(1, int(
                {"passive": 2, "cautious": 5, "aggressive": 12, "alerted": 25}[behavior]
                * self._attacker_profile["noise"]
            )),
            honeypot_probe_count=self._state.honeypot_probe_count,
            # pivot_prob modulates lateral attempt signal
            attacker_lateral_attempts=max(0, int(
                {"passive": 0, "cautious": 1, "aggressive": 3, "alerted": 5}[behavior]
                * self._attacker_profile["pivot_prob"]
            )),
            last_action_outcome=self._state.last_action_outcome,
            attacker_behavior=behavior,
            attacker_profile=self._attacker_profile_name,  # observable — agent knows attacker type
            honeypot_fidelity=round(self._state.honeypot_fidelity, 3),  # agent's own actions built this
            active_alerts=list(self._state.active_alerts),
            available_artifacts=available_arts,
            secured_artifacts=list(self._state.locked_artifacts),
            reward=reward,
            valid_actions=valid,
        )

    def _get_valid_actions(self) -> List[str]:
        """Return list of valid action types for current phase."""
        if self._state.done:
            return []

        always = ["wait_and_track", "emergency_expel"]

        if self._state.phase == "track":
            return ["observe", "mirror_traffic"] + always
        elif self._state.phase == "contain":
            return ["observe", "mirror_traffic", "redirect",
                    "loud_contain", "partial_covert"] + always
        elif self._state.phase == "evidence":
            return ["lock_artifact"] + always
        return always

    # ─── SIEM EMISSION ───────────────────────────────────────────────────

    def _emit_siem(self, severity: str, rule: str, node: str,
                   details: str, action: str = "", outcome: str = "smooth"):
        """Write a SIEM alert to the state log."""
        self._state.siem_log.append({
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "severity": severity,
            "rule": rule,
            "source_node": node,
            "details": details,
            "agent_action": action,
            "outcome": outcome,
            "step": self._state.step_count,
        })

        # Threshold warnings
        if self._state.detection_risk >= 0.60 and not self._state.attacker_alerted:
            if rule != "Elevated Attacker Suspicion":
                self._state.siem_log.append({
                    "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "severity": "HIGH",
                    "rule": "Elevated Attacker Suspicion",
                    "source_node": "",
                    "details": f"Detection risk at {self._state.detection_risk:.2f} — approaching critical threshold.",
                    "agent_action": "",
                    "outcome": "minor_anomaly",
                    "step": self._state.step_count,
                })
