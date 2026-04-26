"""data/scenarios_train.yaml index matches data.py ALL_SCENARIOS ids."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from data import ALL_SCENARIOS


def test_scenarios_train_yaml_matches_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "scenarios_train.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    yids = {s["id"] for s in data["scenarios"]}
    pids = {s.id for s in ALL_SCENARIOS}
    assert yids == pids, (yids - pids, pids - yids)


def test_attacker_profiles_yaml_matches_environment() -> None:
    from environment import ATTACKER_PROFILES

    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "attacker_profiles.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["profiles"]
    assert set(data.keys()) == set(ATTACKER_PROFILES.keys())
    for k, v in ATTACKER_PROFILES.items():
        for field in ("speed", "noise", "pivot_prob", "detection_sensitivity"):
            assert abs(float(data[k][field]) - float(v[field])) < 1e-9, (k, field)
