"""Tests for the pure decide() function (no MLflow server needed). Run: pytest -q test_gate.py"""
import pytest
from eta_mlflow import decide

PROD = dict(mae=10.0, mae_rush=12.0, mae_rain=11.0, p95_ms=2.0)


def cand(**over):
    base = dict(mae=8.0, mae_rush=9.0, mae_rain=8.5, p95_ms=5.0)
    base.update(over)
    return base


def test_clear_win_promotes():
    out = decide(PROD, cand())
    assert out["promote"] is True and all(out["criteria"].values())


def test_identical_models_rejected():
    assert decide(PROD, dict(PROD))["promote"] is False


def test_gain_boundary_just_inside_and_outside():
    assert decide(PROD, cand(mae=9.7))["criteria"]["overall_gain"] is True
    assert decide(PROD, cand(mae=9.71))["criteria"]["overall_gain"] is False
    assert decide(PROD, cand(mae=9.69))["criteria"]["overall_gain"] is True


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=10.0))["promote"] is True
    out = decide(PROD, cand(p95_ms=10.01))
    assert out["promote"] is False and out["criteria"]["latency"] is False


def test_rush_slice_regression_blocks_even_with_big_overall_gain():
    out = decide(PROD, cand(mae=5.0, mae_rush=12.0 * 1.05 + 0.01))
    assert out["criteria"]["overall_gain"] is True
    assert out["criteria"]["slice_rush"] is False and out["promote"] is False
    assert decide(PROD, cand(mae=5.0, mae_rush=12.0 * 1.05))["promote"] is True


def test_rain_slice_regression_blocks():
    out = decide(PROD, cand(mae_rain=11.0 * 1.05 + 0.01))
    assert out["criteria"]["slice_rain"] is False and out["promote"] is False


def test_custom_thresholds():
    assert decide(PROD, cand(mae=8.0), min_rel_gain=0.25)["promote"] is False
    assert decide(PROD, cand(mae=8.0), min_rel_gain=0.20)["promote"] is True
    assert decide(PROD, cand(p95_ms=5.0), max_p95_ms=4.0)["promote"] is False
    assert decide(PROD, cand(mae_rush=13.0), max_slice_regression=0.10)["promote"] is True


def test_worse_candidate_has_negative_gain():
    out = decide(PROD, cand(mae=12.0))
    assert out["details"]["rel_gain"] < 0 and out["promote"] is False


def test_pure_function_does_not_mutate_inputs():
    p, c = dict(PROD), cand()
    snap = (dict(p), dict(c))
    decide(p, c)
    assert (p, c) == snap
