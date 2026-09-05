"""
Unit tests for synthetic maze generation (ml/generate_mazes.py).
Enforces solvable NxN mazes, correct tensor shapes, deterministic seeds,
and shortest-path connectivity via BFS.
"""

import pytest
import numpy as np


def test_generate_single_maze_dimensions_and_solvability():
    """Test that a single generated maze has valid dimensions and is guaranteed solvable."""
    from ml.generate_mazes import generate_maze, solve_maze_bfs

    height, width = 8, 8
    maze_grid, start, goal = generate_maze(height=height, width=width, seed=42)

    # Walls = 1, Passages = 0
    assert maze_grid.shape == (height, width)
    assert start == (0, 0)
    assert goal == (height - 1, width - 1)
    assert maze_grid[start] == 0, "Start cell must be a traversable passage"
    assert maze_grid[goal] == 0, "Goal cell must be a traversable passage"

    # Solve via BFS
    path = solve_maze_bfs(maze_grid, start, goal)
    assert path is not None, "Generated maze must have a valid traversable path from start to goal"
    assert len(path) >= (height + width - 1), "Path length must be at least Manhattan distance"
    assert path[0] == start
    assert path[-1] == goal

    # Verify every step in path is adjacent and in a passage
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Path steps must be orthogonal adjacent neighbors"
        assert maze_grid[r1, c1] == 0
        assert maze_grid[r2, c2] == 0


def test_maze_tensor_encoding_shapes():
    """Test that maze dataset generation produces properly formatted (3, N, N) input and (1, N, N) target tensors."""
    from ml.generate_mazes import generate_dataset

    num_samples = 10
    grid_size = 8
    inputs, targets = generate_dataset(num_samples=num_samples, grid_size=grid_size, seed=123)

    # inputs: (num_samples, 3, grid_size, grid_size)
    # Channel 0: walls (1 = wall, 0 = free)
    # Channel 1: start position mask
    # Channel 2: goal position mask
    assert inputs.shape == (num_samples, 3, grid_size, grid_size)
    assert targets.shape == (num_samples, 1, grid_size, grid_size)

    # Check value ranges
    assert set(np.unique(inputs)).issubset({0.0, 1.0})
    assert set(np.unique(targets)).issubset({0.0, 1.0})

    # Start and goal channels must have exactly one active cell per maze
    for i in range(num_samples):
        assert np.sum(inputs[i, 1]) == 1.0, "Start mask must contain exactly one cell"
        assert np.sum(inputs[i, 2]) == 1.0, "Goal mask must contain exactly one cell"
        assert np.sum(targets[i, 0]) >= 2, "Path must contain at least start and goal cells"


def test_deterministic_seed_reproducibility():
    """Test that generating with the same seed yields identical mazes and paths."""
    from ml.generate_mazes import generate_dataset

    in1, tgt1 = generate_dataset(num_samples=5, grid_size=8, seed=999)
    in2, tgt2 = generate_dataset(num_samples=5, grid_size=8, seed=999)
    in3, tgt3 = generate_dataset(num_samples=5, grid_size=8, seed=1000)

    np.testing.assert_array_equal(in1, in2, "Identical seeds must yield identical input tensors")
    np.testing.assert_array_equal(tgt1, tgt2, "Identical seeds must yield identical target path tensors")
    assert not np.array_equal(in1, in3), "Different seeds should produce different mazes"
