"""
Evaluation and Trace Exporter (ml/evaluate.py).
Evaluates the trained RecurrentLatentMazeSolver checkpoint on a held-out test split
across reasoning effort loops K=1..10.
Exports:
- results/accuracy_by_effort.json & site/data/accuracy_by_effort.json
- results/example_traces.json & site/data/example_traces.json
"""

import os
import sys
import json
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import numpy as np

from ml.generate_mazes import generate_dataset, solve_maze_bfs
from ml.model import RecurrentLatentMazeSolver


def is_path_continuous_solution(pred_binary: np.ndarray, walls: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Checks whether the predicted binary mask forms an unbroken, valid passage
    connecting start to goal without cutting through walls.
    """
    if pred_binary[start] < 0.5 or pred_binary[goal] < 0.5:
        return False
    # Check no wall cell is marked as path
    if np.any((pred_binary > 0.5) & (walls > 0.5)):
        return False

    # Extract all active path cells and verify graph connectivity from start to goal
    H, W = pred_binary.shape
    visited = {start}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        if curr == goal:
            return True
        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if (nr, nc) not in visited and pred_binary[nr, nc] > 0.5 and walls[nr, nc] < 0.5:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def evaluate_model_scaling(
    checkpoint_path: str = "ml/checkpoints/best_model.pt",
    num_test: int = 100,
    grid_size: int = 8,
    seed: int = 999,
) -> Tuple[Dict, Dict]:
    """
    Evaluates trained checkpoint across K=1..10 and generates structured scaling artifacts.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train model first via ml/train.py.")

    checkpoint = torch.load(checkpoint_path)
    model = RecurrentLatentMazeSolver(in_channels=3, hidden_dim=checkpoint.get("hidden_dim", 48))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Generate held-out test mazes
    x_test, y_test = generate_dataset(num_test, grid_size=grid_size, seed=seed)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    scaling_curve = []
    all_predictions_by_k = {}

    import time

    print(f"\nEvaluating {num_test} held-out {grid_size}x{grid_size} mazes across reasoning steps K=1..10:")
    print(f"{'K':<4} | {'Exact Solve Rate':<18} | {'Path IoU':<12} | {'BCE Loss':<10} | {'Measured Latency (ms)'}")
    print("-" * 72)

    with torch.no_grad():
        for k in range(1, 11):
            # Benchmark real inference latency (5 runs over test split)
            # Warmup
            _ = model(x_test_t[:16], k=k)
            t_start = time.perf_counter()
            runs = 5
            for _ in range(runs):
                pred_t = model(x_test_t, k=k)
            t_elapsed = time.perf_counter() - t_start
            latency_per_maze_ms = (t_elapsed / (runs * num_test)) * 1000.0

            pred_np = pred_t.numpy()
            all_predictions_by_k[k] = pred_np

            # Compute metrics
            bce = F.binary_cross_entropy(pred_t, y_test_t).item()

            mask_pred = (pred_np > 0.5).astype(np.float32)
            inter = np.sum(mask_pred * y_test, axis=(1, 2, 3))
            union = np.sum((mask_pred + y_test > 0).astype(np.float32), axis=(1, 2, 3))
            iou = float(np.mean(inter / (union + 1e-6)))

            # Exact solve check: valid continuous path connecting start to goal
            solves = 0
            for i in range(num_test):
                walls_i = x_test[i, 0]
                pred_i = mask_pred[i, 0]
                start = (0, 0)
                goal = (grid_size - 1, grid_size - 1)
                if is_path_continuous_solution(pred_i, walls_i, start, goal):
                    solves += 1

            solve_rate = float(solves / num_test)

            scaling_curve.append({
                "step": k,
                "exact_solve_rate": round(solve_rate, 4),
                "exact_solve_percent": round(solve_rate * 100, 1),
                "path_iou": round(iou, 4),
                "mean_bce_loss": round(bce, 4),
                "measured_latency_ms": round(latency_per_maze_ms, 3),
                "loop_count": k,
            })

            print(f"{k:<4} | {solve_rate * 100:<17.1f}% | {iou:<12.4f} | {bce:<10.4f} | {latency_per_maze_ms:.3f} ms / maze")

    # Build accuracy_by_effort.json
    accuracy_data = {
        "metadata": {
            "model_name": "RecurrentLatentMazeSolver (ConvGRU)",
            "param_count": total_params,
            "test_set_size": num_test,
            "grid_size": [grid_size, grid_size],
            "random_seed": seed,
            "provenance": "PRECOMPUTED_REAL",
            "eval_summary": "Evaluated on held-out synthetic test mazes across reasoning steps K=1..10 with empirical latency benchmarking."
        },
        "scaling_curve": scaling_curve
    }

    # Extract 4 diverse demonstration puzzles for the interactive web UI
    demo_indices = [0, 3, 7, 12] # pre-selected diverse test puzzles
    difficulty_labels = ["Fast Corridor (Direct)", "Medium Detour (Branches)", "Complex Maze (Dead Ends)", "Winding S-Path (Long)"]

    example_puzzles = []
    for idx_pos, idx in enumerate(demo_indices):
        walls = x_test[idx, 0].tolist()
        start = [0, 0]
        goal = [grid_size - 1, grid_size - 1]

        # Extract ground truth path coordinates
        gt_mask = y_test[idx, 0]
        gt_path = solve_maze_bfs(np.array(walls), tuple(start), tuple(goal))

        steps_data = {}
        for k in range(1, 11):
            pred_grid = all_predictions_by_k[k][idx, 0]
            pred_mask = (pred_grid > 0.5).astype(np.float32)

            inter_k = np.sum(pred_mask * gt_mask)
            union_k = np.sum((pred_mask + gt_mask > 0).astype(np.float32))
            iou_k = float(inter_k / (union_k + 1e-6))
            is_solved = is_path_continuous_solution(pred_mask, np.array(walls), tuple(start), tuple(goal))

            # Find active predicted path cells
            predicted_coords = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if pred_grid[r, c] > 0.5:
                        predicted_coords.append([r, c])

            steps_data[str(k)] = {
                "step": k,
                "probabilities": [[round(float(val), 3) for val in row] for row in pred_grid],
                "predicted_path": predicted_coords,
                "iou": round(iou_k, 3),
                "is_solved": is_solved,
            }

        example_puzzles.append({
            "puzzle_id": f"maze_0{idx_pos + 1}",
            "title": f"Puzzle {idx_pos + 1}: {difficulty_labels[idx_pos]}",
            "difficulty": difficulty_labels[idx_pos],
            "grid_size": grid_size,
            "walls": walls,
            "start": start,
            "goal": goal,
            "ground_truth_path": gt_path,
            "ground_truth_mask": [[int(v) for v in row] for row in gt_mask],
            "steps": steps_data,
        })

    traces_data = {
        "provenance": "PRECOMPUTED_REAL",
        "description": "Precomputed model outputs and probability grids recorded once per puzzle per loop count K.",
        "puzzles": example_puzzles
    }

    # Save to results/ and site/data/
    for out_dir in ["results", "site/data"]:
        os.makedirs(out_dir, exist_ok=True)
        acc_path = os.path.join(out_dir, "accuracy_by_effort.json")
        trc_path = os.path.join(out_dir, "example_traces.json")

        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(accuracy_data, f, indent=2)

        with open(trc_path, "w", encoding="utf-8") as f:
            json.dump(traces_data, f, indent=2)

        print(f"Exported artifacts to {acc_path} and {trc_path}")

    return accuracy_data, traces_data


if __name__ == "__main__":
    evaluate_model_scaling()
