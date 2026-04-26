"""
ShadowNet — Evaluation Harness
════════════════════════════════════════════════════════════════════
Runs baseline + random agents across all tasks × all attacker profiles.
Outputs a Markdown table and saves JSON to training/eval_baseline.json.

Usage:
    python eval_harness.py
    python eval_harness.py --seed 42   # reproducible run
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from environment import ShadowNetEnv, ATTACKER_PROFILES
from agent import BaselineAgent, RandomAgent
from grader import grade_episode


def run_agent_on_profile(
    agent_cls,
    task_name: str,
    profile_name: str,
    n_episodes: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """
    Run an agent on a specific task + attacker profile.
    Forces the environment to use the given attacker profile for every episode
    by overriding the sampled profile after reset.
    """
    scores: List[float] = []
    collapses: int = 0

    for i in range(n_episodes):
        env = ShadowNetEnv()
        ep_seed = (seed + i) if seed is not None else None
        obs = env.reset(task_name, scenario_index=i % 5, seed=ep_seed)

        # Force attacker profile (bypasses random sampling)
        env._attacker_profile_name = profile_name
        env._attacker_profile = ATTACKER_PROFILES[profile_name]

        agent = agent_cls()
        if hasattr(agent, "reset"):
            agent.reset()

        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)

        grade = grade_episode(env.state)
        scores.append(grade["score"])
        if env.state.attacker_alerted:
            collapses += 1

    avg = round(sum(scores) / len(scores), 3) if scores else 0.0
    return {
        "avg_score": avg,
        "scores": scores,
        "collapse_rate": round(collapses / n_episodes, 2),
        "episodes": n_episodes,
    }


def run_full_harness(seed: Optional[int] = None) -> Dict:
    """Run both agents on all tasks × all profiles. Returns nested result dict."""
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]
    profiles = list(ATTACKER_PROFILES.keys())  # stealthy, aggressive, noisy, adaptive
    agents = {"baseline": BaselineAgent, "random": RandomAgent}

    results: Dict = {}

    for agent_name, agent_cls in agents.items():
        results[agent_name] = {}
        for task in tasks:
            results[agent_name][task] = {}
            for profile in profiles:
                print(f"  Running {agent_name} | {task} | {profile}...", end=" ", flush=True)
                r = run_agent_on_profile(agent_cls, task, profile, n_episodes=5, seed=seed)
                results[agent_name][task][profile] = r
                print(f"avg={r['avg_score']} collapse={r['collapse_rate']}")

    return results


def print_markdown_table(results: Dict) -> str:
    """Print a Markdown table comparing baseline vs random per task/profile."""
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]
    profiles = list(ATTACKER_PROFILES.keys())

    lines = []
    lines.append("## Evaluation Results\n")
    lines.append("### Score (avg over 5 episodes per cell)\n")
    lines.append("| Task | Profile | Random Score | Baseline Score | Baseline Collapse Rate |")
    lines.append("|------|---------|-------------|---------------|----------------------|")

    for task in tasks:
        for profile in profiles:
            r_rand = results.get("random", {}).get(task, {}).get(profile, {})
            r_base = results.get("baseline", {}).get(task, {}).get(profile, {})
            lines.append(
                f"| {task} | {profile} "
                f"| {r_rand.get('avg_score', '?')} "
                f"| {r_base.get('avg_score', '?')} "
                f"| {r_base.get('collapse_rate', '?')} |"
            )

    table = "\n".join(lines)
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser(description="ShadowNet Evaluation Harness")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs")
    parser.add_argument("--out", type=str, default="training/eval_baseline.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("=" * 64)
    print("ShadowNet Evaluation Harness")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 64)
    print()

    results = run_full_harness(seed=args.seed)

    # Print Markdown table
    print()
    table = print_markdown_table(results)

    # Save JSON
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Also save the Markdown table
    md_out = args.out.replace(".json", "_table.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"Markdown table saved to {md_out}")


if __name__ == "__main__":
    main()
