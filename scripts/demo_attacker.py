"""
Demo: run a single episode and print attacker path + defender actions step by step.

Usage (from repo root):
  python scripts/demo_attacker.py
  python scripts/demo_attacker.py --task shadow-hard --profile aggressive_apt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import BaselineAgent
from environment import ShadowNetEnv, ATTACKER_PROFILES
from grader import grade_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo attacker path + defender actions.")
    parser.add_argument("--task", default="shadow-easy")
    parser.add_argument("--profile", default=None, choices=list(ATTACKER_PROFILES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = ShadowNetEnv()
    obs = env.reset(args.task, seed=args.seed)

    if args.profile and args.profile in ATTACKER_PROFILES:
        env._attacker_profile_name = args.profile
        env._attacker_profile = ATTACKER_PROFILES[args.profile]

    agent = BaselineAgent()
    agent.reset()

    print("=" * 60)
    print(f"  ShadowNet Demo -- {args.task}")
    print(f"  Attacker profile: {env._attacker_profile_name}")
    print(f"  Scenario: {obs.task_name}")
    print("=" * 60)

    step = 0
    while not obs.done:
        step += 1
        action = agent.act(obs)

        print(f"\n--- Step {step} | Phase: {obs.phase} ---")
        print(f"  Anomalous nodes : {obs.anomalous_nodes}")
        print(f"  Attacker behavior: {obs.attacker_behavior}")
        print(f"  Query frequency  : {obs.attacker_query_frequency}")
        print(f"  Lateral attempts : {obs.attacker_lateral_attempts}")
        print(f"  Honeypot fidelity: {obs.honeypot_fidelity:.2f}")
        print(f"  >> ACTION: {action.action_type.value} {action.target or ''}")

        obs = env.step(action)
        print(f"  Outcome: {obs.last_action_outcome}")

    grade = grade_episode(env.state)
    print("\n" + "=" * 60)
    print("  Episode Complete")
    print(f"  Final score: {grade['score']}")
    print(f"  Components : {grade['components']}")
    print(f"  Alerted?   : {env.state.attacker_alerted}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
