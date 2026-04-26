"""
Quick health check: one baseline episode, all-task averages, reward clamp.
Run from repo root:  python scripts/verify_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import ShadowNetEnv
from agent import run_baseline_episode, run_baseline_all_tasks
from grader import compute_reward
from models import SREState


def main() -> int:
    # Single episode
    env = ShadowNetEnv()
    result = run_baseline_episode(env, "shadow-easy")
    print(f"Easy score: {result['score']}")
    print(f"Components: {result['components']}")
    print(f"Metadata: {result['metadata']}")
    print()

    # All tasks
    env2 = ShadowNetEnv()
    all_results = run_baseline_all_tasks(env2)
    for task, data in all_results.items():
        print(f"{task}: avg={data['avg_score']}, scores={data['scores']}")

    # Reward clamp edge cases
    empty = SREState()
    print(f"\nEmpty state reward: {compute_reward(empty)} (should be ~0.5-0.9)")
    empty.real_data_accessed = 100
    empty.total_real_assets = 1
    empty.peak_detection_risk = 1.0
    empty.artifacts_secured = 0
    empty.total_artifacts = 10
    print(
        f"Worst case reward: {compute_reward(empty)} "
        f"(should be >0.01 and <0.20 — clamp enforced)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
