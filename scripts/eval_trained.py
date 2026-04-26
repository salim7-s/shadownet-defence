"""
Evaluate a trained HF/PEFT checkpoint against the same ShadowNet benchmark grid.

Usage:
  python scripts/eval_trained.py --checkpoint shadownet-grpo-adapters --base-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import ATTACKER_PROFILES
from training.prompts import build_chat_messages
from training.reward_adapter import EpisodeRunner


def _load_generator(checkpoint: str, base_model: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - only hit on machines without optional deps
        raise RuntimeError(
            "transformers/torch are required for eval_trained.py. Install training deps first."
        ) from exc

    tokenizer_source = base_model or checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    model = None
    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=getattr(torch, "float16", None),
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=getattr(torch, "float16", None),
            device_map="auto",
        )

    model.eval()

    def generate_fn(system_prompt: str, user_prompt: str, _obs) -> str:
        messages = build_chat_messages(_obs)
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = f"{system_prompt}\n\n{user_prompt}\n"

        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return completion.strip()

    return generate_fn


def _evaluate_checkpoint(
    generate_fn,
    *,
    seed: int,
    episodes: int,
) -> dict[str, Any]:
    runner = EpisodeRunner()
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]
    profiles = list(ATTACKER_PROFILES.keys())

    results: dict[str, Any] = {"trained": {}}
    for task in tasks:
        results["trained"][task] = {}
        for profile in profiles:
            bucket = []
            for i in range(episodes):
                result = runner.run_episode(
                    generate_fn,
                    task_name=task,
                    seed=seed + i,
                    scenario_index=i % 5,
                    attacker_profile=profile,
                )
                bucket.append(result)

            avg_score = round(sum(x["score"] for x in bucket) / len(bucket), 3)
            avg_combined = round(sum(x["combined_reward"] for x in bucket) / len(bucket), 3)
            collapse_rate = round(sum(1 for x in bucket if x["collapse"]) / len(bucket), 2)
            parse_failures = sum(x["parse_failures"] for x in bucket)

            results["trained"][task][profile] = {
                "avg_score": avg_score,
                "avg_combined_reward": avg_combined,
                "collapse_rate": collapse_rate,
                "parse_failures": parse_failures,
                "episodes": episodes,
            }

    return results


def _build_markdown(trained_results: dict[str, Any], baseline_results: dict[str, Any] | None) -> str:
    tasks = ["shadow-easy", "shadow-medium", "shadow-hard"]
    profiles = list(ATTACKER_PROFILES.keys())

    lines = [
        "## Trained Evaluation",
        "",
        "| Task | Profile | Baseline | Trained | Collapse | Parse Failures |",
        "|------|---------|----------|---------|----------|----------------|",
    ]
    for task in tasks:
        for profile in profiles:
            trained = trained_results["trained"][task][profile]
            baseline = ""
            if baseline_results:
                baseline = str(baseline_results.get("baseline", {}).get(task, {}).get(profile, {}).get("avg_score", "?"))
            else:
                baseline = "n/a"
            lines.append(
                f"| {task} | {profile} | {baseline} | {trained['avg_score']} | "
                f"{trained['collapse_rate']} | {trained['parse_failures']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained ShadowNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--baseline-json", default="training/eval_baseline.json")
    args = parser.parse_args()

    generate_fn = _load_generator(args.checkpoint, args.base_model)
    trained_results = _evaluate_checkpoint(generate_fn, seed=args.seed, episodes=args.episodes)

    baseline_results = None
    baseline_path = Path(args.baseline_json)
    if baseline_path.exists():
        baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / "eval_summary.json"
    summary_md = out_dir / "eval_summary.md"
    summary_json.write_text(json.dumps(trained_results, indent=2), encoding="utf-8")
    summary_md.write_text(_build_markdown(trained_results, baseline_results), encoding="utf-8")

    print(f"Saved JSON to {summary_json}")
    print(f"Saved Markdown to {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
