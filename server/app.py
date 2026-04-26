"""
ShadowNet — FastAPI Server (OpenEnv Native)
═══════════════════════════════════════════════════════════════
Uses OpenEnv's create_app() for automatic endpoint generation
+ custom endpoints for dashboard and SIEM feed.
Port 7860 for HuggingFace Spaces deployment.

CRITICAL FIX: All endpoints share a SINGLE _env instance so
the dashboard's /reset, /step, /network-state and /siem-alerts
all reflect the same episode state.
═══════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from models import SREAction, SREObservation, ActionType
from environment import ShadowNetEnv
from grader import grade_episode, compute_reward
from agent import BaselineAgent, run_baseline_episode, run_baseline_all_tasks
from data import TASK_METADATA

from inference import format_structured_decision_log as _build_decision_log

# ─── SINGLE SHARED ENVIRONMENT INSTANCE ─────────────────────────────────────
# ALL endpoints (reset, step, state, network-state, siem-alerts) use this ONE
# instance so the dashboard graph reflects the actual current episode.
_env = ShadowNetEnv()


def _has_route(path: str, method: str) -> bool:
    """Return True when the current FastAPI app already exposes path/method."""
    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", set()) or set()
        if method.upper() in methods:
            return True
    return False

# ─── CREATE APP: OpenEnv native if available, else manual ────────────────────

try:
    from openenv.core.env_server import create_app

    app = create_app(
        env=lambda: _env,
        action_cls=SREAction,
        observation_cls=SREObservation,
        env_name="shadownet",
        max_concurrent_envs=64,
    )
    _msg = "[ShadowNet] Using OpenEnv create_app - native framework integration"
    sys.stdout.buffer.write((_msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()

except (ImportError, Exception) as e:
    _msg = f"[ShadowNet] OpenEnv create_app not available ({e}), using manual endpoints"
    sys.stdout.buffer.write((_msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()
    app = FastAPI(
        title="ShadowNet: Deceptive Containment Environment",
        description="OpenEnv RL environment for training AI agents in active cyber defense.",
        version="1.0.0",
    )

    class ResetRequest(BaseModel):
        task_name: str = "shadow-easy"
        scenario_index: Optional[int] = None
        seed: Optional[int] = None

    class StepRequest(BaseModel):
        action: dict

    @app.get("/health")
    def health():
        return {"status": "ok", "environment": "shadownet", "version": "1.0.0"}

    @app.get("/tasks")
    def get_tasks():
        return {"tasks": TASK_METADATA}

    @app.post("/reset")
    def reset_env(req: ResetRequest):
        obs = _env.reset(req.task_name, req.scenario_index, seed=req.seed)
        return obs.model_dump()

    @app.post("/step")
    def step_env(req: StepRequest):
        action_data = req.action
        action_type_str = action_data.get("action_type", "wait_and_track")
        target = action_data.get("target", None)
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.WAIT_AND_TRACK
        action = SREAction(action_type=action_type, target=target)
        obs = _env.step(action)
        # §9: Populate decision_log with structured reasoning for every step
        obs.decision_log = _build_decision_log(obs, action)
        return obs.model_dump()

    @app.get("/state")
    def get_state():
        return _env.state.model_dump()


# ─── SHARED ADDITIONAL ENDPOINTS ─────────────────────────────────────────────
# These always use the same _env instance regardless of OpenEnv mode.

if not _has_route("/tasks", "GET"):
    @app.get("/tasks")
    def get_tasks():
        """Task registry used by the OpenEnv manifest and HF Spaces judges."""
        return {"tasks": TASK_METADATA}


@app.get("/grader")
def get_grader():
    """Run baseline agent on all tasks and return scores."""
    env2 = ShadowNetEnv()
    results = run_baseline_all_tasks(env2)
    return {"grader_results": results}


@app.get("/baseline")
def get_baseline():
    """Baseline agent metrics across all tasks."""
    env2 = ShadowNetEnv()
    results = run_baseline_all_tasks(env2)
    table = []
    for task, data in results.items():
        table.append({
            "task": task,
            "avg_score": data["avg_score"],
            "episodes": data["episodes_run"],
            "individual_scores": data["scores"],
        })
    return {
        "agent_type": "rule-based baseline",
        "description": "Deterministic observe→mirror→redirect strategy with no ML",
        "results": table,
    }


@app.get("/network-state")
def get_network_state():
    """
    Node states for the live network graph SVG.
    Uses the same _env as /reset and /step — reflects current episode.
    """
    state = _env.state
    scenario = _env._scenario

    if not scenario:
        # Not yet reset — return a helpful empty state
        return {"nodes": [], "edges": [], "phase": "idle", "step": 0,
                "honeypot_fidelity": 0.85,
                "message": "No episode active. Click Reset to start."}

    nodes = []
    for node in scenario.nodes:
        name = node.name
        if name in state.compromised_nodes:
            s = "compromised"
        elif name in state.honeypot_nodes:
            s = "honeypot"
        elif name in state.mirroring_nodes:
            s = "mirroring"
        elif name in state.probed_nodes:
            s = "probed"
        else:
            s = "clean"
        nodes.append({
            "id": name,
            "label": name,
            "state": s,
            "sensitivity": node.sensitivity,
            "attacker_here": (name == state.attacker_current_node),
            "x": node.x,
            "y": node.y,
        })

    edges = [{"from": e.source, "to": e.target} for e in scenario.edges]

    return {
        "nodes": nodes,
        "edges": edges,
        "phase": state.phase,
        "step": state.step_count,
        "attacker_profile": _env._attacker_profile_name,
        # detection_risk intentionally omitted — hidden from agent (partial observability)
        "honeypot_fidelity": round(state.honeypot_fidelity, 3),
        "attacker_current_node": state.attacker_current_node,
    }


@app.get("/siem-alerts")
def get_siem_alerts():
    """Last 50 SIEM events for the live alert feed — same _env as /step."""
    return {
        "alerts": _env.state.siem_log[-50:],
        "total": len(_env.state.siem_log),
        "episode_id": _env.state.episode_id,
        "phase": _env.state.phase,
    }



# ─── REASONING LOG ────────────────────────────────────────────────────────────
# A thread-safe deque of the last 50 structured reasoning entries.
# Populated on every /step call and by demo runners.

import collections, threading
_reasoning_lock = threading.Lock()
_reasoning_log: collections.deque = collections.deque(maxlen=50)


def _append_reasoning(step: int, obs: "SREObservation", action: "SREAction"):
    """Append one OBSERVED/INFERRED/ACTION/REASON entry to the reasoning deque."""
    entry = {
        "step": step,
        "observed": {
            "phase": obs.phase,
            "behavior": obs.attacker_behavior,
            "query_freq": obs.attacker_query_frequency,
            "probe_count": obs.honeypot_probe_count,
            "lateral": obs.attacker_lateral_attempts,
            "outcome": obs.last_action_outcome,
            "fidelity": obs.honeypot_fidelity,
        },
        "inferred": _build_decision_log(obs, action).split("\n")[1].replace("INFERRED: ", ""),
        "action": f"{action.action_type.value}" + (f" {action.target}" if action.target else ""),
        "reason": _build_decision_log(obs, action).split("\n")[3].replace("REASON: ", ""),
    }
    with _reasoning_lock:
        _reasoning_log.append(entry)


@app.get("/reasoning-log")
def get_reasoning_log():
    """Last 20 structured OBSERVED/INFERRED/ACTION/REASON reasoning entries from live agent."""
    with _reasoning_lock:
        entries = list(_reasoning_log)[-20:]
    return {
        "entries": entries,
        "total": len(entries),
        "episode_id": _env.state.episode_id,
    }


# ─── DEMO ENDPOINTS ───────────────────────────────────────────────────────────
# Seeded, deterministic episode replays for the judging demo.
# /demo/run-bad  → aggressive, non-stealthy policy → triggers collapse
# /demo/run-good → proper mirror→redirect→lock policy → clean success

def _run_demo_bad(seed: int = 7) -> dict:
    """
    Bad-strategy demo: redirects immediately without mirroring, triggers detection.
    Returns full trajectory with per-step reward components.
    """
    from agent import RandomAgent
    import random as _rand

    demo_env = ShadowNetEnv()
    obs = demo_env.reset(task_name="shadow-easy", scenario_index=0, seed=seed)

    trajectory = []
    step = 0

    while not obs.done and step < 20:
        step += 1
        # Bad policy: always try to redirect (or loud_contain) without mirroring first
        if "redirect" in obs.valid_actions and obs.anomalous_nodes:
            action = SREAction(action_type=ActionType.REDIRECT, target=obs.anomalous_nodes[0])
        elif "loud_contain" in obs.valid_actions and obs.anomalous_nodes:
            action = SREAction(action_type=ActionType.LOUD_CONTAIN, target=obs.anomalous_nodes[0])
        elif "observe" in obs.valid_actions and obs.network_nodes:
            action = SREAction(action_type=ActionType.OBSERVE, target=obs.network_nodes[0])
        else:
            action = SREAction(action_type=ActionType.WAIT_AND_TRACK)

        prev_obs = obs
        obs = demo_env.step(action)
        _append_reasoning(step, prev_obs, action)

        trajectory.append({
            "step": step,
            "action": f"{action.action_type.value}" + (f" {action.target}" if action.target else ""),
            "outcome": obs.last_action_outcome,
            "behavior": obs.attacker_behavior,
            "alerted": obs.attacker_behavior == "alerted",
            "phase": obs.phase,
        })

    from grader import grade_episode, compute_reward_components
    grade = grade_episode(demo_env.state)
    return {
        "strategy": "bad",
        "seed": seed,
        "final_score": grade["score"],
        "outcome": grade["outcome"],
        "components": grade["components"],
        "collapse": demo_env.state.attacker_alerted,
        "steps": step,
        "trajectory": trajectory,
    }


def _run_demo_good(seed: int = 7) -> dict:
    """
    Good-strategy demo: observe → mirror → wait → redirect → lock artifacts.
    Returns full trajectory showing clean containment.
    """
    from agent import BaselineAgent

    demo_env = ShadowNetEnv()
    obs = demo_env.reset(task_name="shadow-easy", scenario_index=0, seed=seed)

    agent = BaselineAgent()
    agent.reset()

    trajectory = []
    step = 0

    while not obs.done and step < 20:
        step += 1
        action = agent.act(obs)
        prev_obs = obs
        obs = demo_env.step(action)
        _append_reasoning(step, prev_obs, action)

        trajectory.append({
            "step": step,
            "action": f"{action.action_type.value}" + (f" {action.target}" if action.target else ""),
            "outcome": obs.last_action_outcome,
            "behavior": obs.attacker_behavior,
            "alerted": obs.attacker_behavior == "alerted",
            "phase": obs.phase,
        })

    from grader import grade_episode
    grade = grade_episode(demo_env.state)
    return {
        "strategy": "good",
        "seed": seed,
        "final_score": grade["score"],
        "outcome": grade["outcome"],
        "components": grade["components"],
        "collapse": demo_env.state.attacker_alerted,
        "steps": step,
        "trajectory": trajectory,
    }


class DemoRequest(BaseModel):
    seed: int = 7


@app.post("/demo/run-bad")
def demo_run_bad(req: DemoRequest = DemoRequest()):
    """Run the bad-strategy demo episode (deterministic). Triggers detection collapse."""
    return _run_demo_bad(seed=req.seed)


@app.post("/demo/run-good")
def demo_run_good(req: DemoRequest = DemoRequest()):
    """Run the good-strategy demo episode (deterministic). Shows clean containment."""
    return _run_demo_good(seed=req.seed)


@app.get("/demo/compare")
def demo_compare():
    """Run both strategies side-by-side and return the comparison."""
    bad = _run_demo_bad(seed=7)
    good = _run_demo_good(seed=7)
    return {
        "bad_strategy": {
            "score": bad["final_score"],
            "outcome": bad["outcome"],
            "collapse": bad["collapse"],
            "components": bad["components"],
        },
        "good_strategy": {
            "score": good["final_score"],
            "outcome": good["outcome"],
            "collapse": good["collapse"],
            "components": good["components"],
        },
        "delta": round(good["final_score"] - bad["final_score"], 3),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Interactive crisis dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>ShadowNet Dashboard — index.html not found</h1>")


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

