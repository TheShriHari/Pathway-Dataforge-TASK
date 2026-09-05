"""
Unit tests for evaluation pipeline and scaling curve characteristics (ml/evaluate.py).
Verifies:
- Monotonic-or-plateauing accuracy trend across K (K=10 >= K=1)
- Diminishing returns (early gains exceed late gains)
- JSON export schema and metadata compliance
"""

import os
import json
import pytest
import numpy as np


def test_evaluation_schema_accuracy_by_effort():
    """Verify that results/accuracy_by_effort.json adheres to the expected contract."""
    from ml.evaluate import evaluate_model_scaling

    # Test evaluation function on mock data or real function
    test_file = "site/data/accuracy_by_effort.json"
    if not os.path.exists(test_file):
        pytest.fail(f"Required artifact {test_file} has not been produced yet.")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["provenance"] == "PRECOMPUTED_REAL"
    assert "scaling_curve" in data
    curve = data["scaling_curve"]

    assert len(curve) == 10, "Scaling curve must contain entries for steps K=1 to K=10"
    for idx, item in enumerate(curve):
        assert item["step"] == idx + 1
        assert 0.0 <= item["exact_solve_rate"] <= 1.0
        assert 0.0 <= item["path_iou"] <= 1.0
        assert "mean_bce_loss" in item


def test_evaluation_monotonicity_and_diminishing_returns():
    """
    Verify empirical scaling properties:
    1. High-K accuracy (K=10) >= Low-K accuracy (K=1) within tolerance.
    2. Diminishing returns: Gain from K=1 to K=5 is greater than gain from K=5 to K=10.
    """
    test_file = "site/data/accuracy_by_effort.json"
    if not os.path.exists(test_file):
        pytest.fail(f"Required artifact {test_file} has not been produced yet.")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    curve = data["scaling_curve"]
    ious = [pt["path_iou"] for pt in curve]
    exact_solves = [pt["exact_solve_rate"] for pt in curve]

    # Property 1: Higher K solves better than K=1 (both on exact solve rate and path IoU)
    exact_gain = exact_solves[4] - exact_solves[0]
    iou_gain = ious[4] - ious[0]
    assert exact_gain >= 0.10, (
        f"K=5 exact solve rate ({exact_solves[4]:.3f}) should exceed K=1 ({exact_solves[0]:.3f}) by at least +0.10, got {exact_gain:.3f}"
    )
    assert ious[-1] >= ious[0] + 0.03, (
        f"K=10 IoU ({ious[-1]:.3f}) should exceed K=1 IoU ({ious[0]:.3f}) by at least +0.03"
    )

    # Property 2: Concavity / diminishing returns (early gain exceeds late gain)
    early_exact_gain = exact_solves[4] - exact_solves[0] # Steps 1 to 5
    late_exact_gain = exact_solves[9] - exact_solves[4]  # Steps 5 to 10
    assert early_exact_gain > late_exact_gain, (
        f"Exact solve gains should show diminishing returns: early gain ({early_exact_gain:.3f}) vs late gain ({late_exact_gain:.3f})"
    )


def test_example_traces_schema_and_integrity():
    """Verify results/example_traces.json contains valid puzzles and per-K predictions."""
    test_file = "site/data/example_traces.json"
    if not os.path.exists(test_file):
        pytest.fail(f"Required artifact {test_file} has not been produced yet.")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["provenance"] == "PRECOMPUTED_REAL"
    assert "puzzles" in data
    assert len(data["puzzles"]) >= 3, "Should provide at least 3 example demonstration puzzles"

    for puzzle in data["puzzles"]:
        assert "puzzle_id" in puzzle
        assert "walls" in puzzle
        assert "ground_truth_path" in puzzle
        assert "steps" in puzzle
        assert len(puzzle["steps"]) == 10
        for k_str in ["1", "5", "10"]:
            assert k_str in puzzle["steps"]
            step_data = puzzle["steps"][k_str]
            assert "probabilities" in step_data
            assert "is_solved" in step_data
            assert "iou" in step_data
