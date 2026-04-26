"""
Build JSONL SFT data from rule-based expert (BaselineAgent) rollouts.

Each line is a chat sample:
  system   -> training system prompt (ACTION/TARGET/REASON format)
  user     -> structured observation (same format used by eval_trained.py)
  assistant -> ACTION: .../TARGET: .../REASON: ... (matches parse_action_output)

Usage (from repo root):
  python training/build_sft_dataset.py -o training/sft_shadownet.jsonl
  python training/build_sft_dataset.py -o training/sft_high_reward.jsonl --min-score 0.55
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import BaselineAgent
from grader import grade_episode
from environment import ShadowNetEnv, ATTACKER_PROFILES
from training.prompts import SYSTEM_PROMPT, build_observation_prompt, format_action_block


def _make_reason(obs, action) -> str:
    """Generate a brief reason string for the expert action."""
    a = action.action_type.value
    if a == "wait_and_track":
        return "Letting detection risk decay before next containment move."
    if a == "mirror_traffic":
        return "Raising honeypot fidelity before redirect to reduce detection risk."
    if a == "redirect":
        return "Fidelity is high enough; redirecting attacker into honeypot."
    if a == "lock_artifact":
        return "Locking forensic artifact before it decays."
    if a == "observe":
        return "Gathering visibility on anomalous node."
    if a == "emergency_expel":
        return "Emergency expel -- data at imminent risk."
    return "Standard containment move for current phase and signals."


def _rollout_one_episode(
    task: str,
    scenario_index: int,
    seed: int | None,
    profile: str | None,
) -> tuple[list[dict], float]:
    """Returns (list of message-dict training rows, final score)."""
    env = ShadowNetEnv()
    obs = env.reset(task, scenario_index=scenario_index, seed=seed)
    if profile is not None and profile in ATTACKER_PROFILES:
        env._attacker_profile_name = profile
        env._attacker_profile = ATTACKER_PROFILES[profile]

    agent = BaselineAgent()
    agent.reset()
    rows: list[dict] = []

    while not obs.done:
        action = agent.act(obs)
        reason = _make_reason(obs, action)
        assistant = format_action_block(action, reason=reason)
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_observation_prompt(obs)},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "task": task,
                "scenario_index": scenario_index,
                "step": obs.step,
                "attacker_profile": env._attacker_profile_name,
            },
        }
        rows.append(row)
        obs = env.step(action)

    grade = grade_episode(env.state)
    return rows, float(grade["score"])


def main() -> int:
    p = argparse.ArgumentParser(description="Build SFT JSONL from BaselineAgent rollouts.")
    p.add_argument(
        "-o", "--output",
        default="training/sft_shadownet.jsonl",
        help="Output JSONL path (default: training/sft_shadownet.jsonl)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="If set, drop full episodes with final grader score below this (e.g. 0.55).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat each (task, scenario_index) with different seeds.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for scenario rotation.",
    )
    args = p.parse_args()
    random.seed(args.seed)

    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]
    profiles = list(ATTACKER_PROFILES.keys())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    total_eps = 0
    kept_eps = 0

    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            for scenario_index in range(5):
                for r in range(args.repeats):
                    ep_seed = random.randint(0, 1_000_000)
                    prof = profiles[(scenario_index + r + hash(task) % 7) % len(profiles)]
                    rows, score = _rollout_one_episode(
                        task, scenario_index, ep_seed, prof,
                    )
                    total_eps += 1
                    if args.min_score is not None and score < args.min_score:
                        continue
                    kept_eps += 1
                    for row in rows:
                        row["metadata"]["episode_score"] = round(score, 4)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_steps += 1

    print(f"Wrote {out_path}  episodes_seen={total_eps}  episodes_kept={kept_eps}  training_rows={total_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
