"""
Regenerate baseline evaluation artifacts from code.

Usage:
  python scripts/eval_baseline.py --seed 42 --out training/eval_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_harness import print_markdown_table, run_full_harness


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate ShadowNet baseline artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="training/eval_baseline.json")
    args = parser.parse_args()

    results = run_full_harness(seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table = print_markdown_table(results)
    md_out = out_path.with_name(out_path.stem + "_table.md")
    md_out.write_text(table + "\n", encoding="utf-8")

    print(f"\nSaved JSON to {out_path}")
    print(f"Saved table to {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
