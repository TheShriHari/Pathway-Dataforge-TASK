"""
Synthetic maze generator and BFS path solver for the latent reasoning pipeline.
Produces deterministic NxN mazes guaranteed to be solvable, with 3-channel input
encodings and binary shortest-path target masks.
"""

from collections import deque
from typing import List, Optional, Tuple
import numpy as np


def generate_maze(
    height: int = 8,
    width: int = 8,
    seed: Optional[int] = None,
    loop_prob: float = 0.15,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Generate an HxW grid maze where 0 = passage and 1 = wall.
    Start is at (0, 0) and goal is at (height - 1, width - 1).
    Guarantees a valid traversable passage connecting start to goal.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    start = (0, 0)
    goal = (height - 1, width - 1)

    # Initialize all cells as walls (1)
    grid = np.ones((height, width), dtype=np.float32)

    # Carve a random self-avoiding walk from start to goal to guarantee solvability
    # Then randomly carve branches and loops
    curr = start
    grid[curr] = 0.0

    # Carve guaranteed path to goal using random walk with directional bias
    path = [curr]
    visited = {curr}
    while curr != goal:
        r, c = curr
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                neighbors.append((nr, nc))

        if not neighbors:
            # Backtrack if stuck
            path.pop()
            if not path:
                # Reset if path collapsed
                curr = start
                path = [curr]
                visited = {curr}
                grid = np.ones((height, width), dtype=np.float32)
                grid[curr] = 0.0
                continue
            curr = path[-1]
            continue

        # Prefer moves that get closer to goal with 60% probability
        dist_to_goal = [abs(nr - goal[0]) + abs(nc - goal[1]) for nr, nc in neighbors]
        min_dist = min(dist_to_goal)
        good_neighbors = [neighbors[i] for i in range(len(neighbors)) if dist_to_goal[i] == min_dist]

        if rng.rand() < 0.65 and good_neighbors:
            next_cell = good_neighbors[rng.randint(len(good_neighbors))]
        else:
            next_cell = neighbors[rng.randint(len(neighbors))]

        grid[next_cell] = 0.0
        visited.add(next_cell)
        path.append(next_cell)
        curr = next_cell

    # Carve additional random branches to create deceptive paths and dead ends
    all_open_cells = list(visited)
    for _ in range(int(height * width * 0.35)):
        if not all_open_cells:
            break
        base_cell = all_open_cells[rng.randint(len(all_open_cells))]
        r, c = base_cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 1.0:
                if rng.rand() < 0.45:
                    grid[nr, nc] = 0.0
                    all_open_cells.append((nr, nc))

    # Occasionally add loops between open cells
    for r in range(height):
        for c in range(width):
            if grid[r, c] == 1.0:
                open_adj = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 0.0:
                        open_adj += 1
                if open_adj >= 2 and rng.rand() < loop_prob:
                    grid[r, c] = 0.0

    # Ensure start and goal are always passages
    grid[start] = 0.0
    grid[goal] = 0.0

    return grid, start, goal


def solve_maze_bfs(
    maze_grid: np.ndarray,
    start: Tuple[int, int] = (0, 0),
    goal: Optional[Tuple[int, int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    Breadth-First Search to find the shortest path from start to goal.
    Returns list of coordinates [(r, c), ...] along the shortest path, or None if no path.
    """
    height, width = maze_grid.shape
    if goal is None:
        goal = (height - 1, width - 1)

    if maze_grid[start] != 0.0 or maze_grid[goal] != 0.0:
        return None

    queue = deque([start])
    visited = {start: None} # child -> parent mapping

    while queue:
        curr = queue.popleft()
        if curr == goal:
            break

        r, c = curr
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and maze_grid[nr, nc] == 0.0:
                neighbor = (nr, nc)
                if neighbor not in visited:
                    visited[neighbor] = curr
                    queue.append(neighbor)

    if goal not in visited:
        return None

    # Reconstruct path from goal back to start
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = visited[curr]
    path.reverse()
    return path


def generate_dataset(
    num_samples: int,
    grid_size: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset of mazes and their optimal shortest path masks.
    Returns:
      inputs: (num_samples, 3, grid_size, grid_size)
              channel 0: walls (1.0 = wall, 0.0 = passage)
              channel 1: start mask (1.0 at start, 0.0 elsewhere)
              channel 2: goal mask (1.0 at goal, 0.0 elsewhere)
      targets: (num_samples, 1, grid_size, grid_size)
              channel 0: binary mask of cells on the optimal shortest path
    """
    inputs = np.zeros((num_samples, 3, grid_size, grid_size), dtype=np.float32)
    targets = np.zeros((num_samples, 1, grid_size, grid_size), dtype=np.float32)

    rng = np.random.RandomState(seed)

    for i in range(num_samples):
        sample_seed = int(rng.randint(0, 1_000_000_000))
        grid, start, goal = generate_maze(height=grid_size, width=grid_size, seed=sample_seed)
        path = solve_maze_bfs(grid, start, goal)

        # Retry if maze was somehow unsolvable (failsafe)
        attempts = 0
        while path is None and attempts < 10:
            sample_seed = int(rng.randint(0, 1_000_000_000))
            grid, start, goal = generate_maze(height=grid_size, width=grid_size, seed=sample_seed)
            path = solve_maze_bfs(grid, start, goal)
            attempts += 1

        if path is None:
            raise RuntimeError(f"Failed to generate solvable maze after 10 attempts at index {i}")

        # Input tensor
        inputs[i, 0] = grid # walls
        inputs[i, 1, start[0], start[1]] = 1.0 # start
        inputs[i, 2, goal[0], goal[1]] = 1.0 # goal

        # Target mask
        for r, c in path:
            targets[i, 0, r, c] = 1.0

    return inputs, targets


if __name__ == "__main__":
    in_arr, tgt_arr = generate_dataset(num_samples=5, grid_size=8, seed=42)
    print(f"Generated dataset shapes: inputs={in_arr.shape}, targets={tgt_arr.shape}")
    print("Sample 0 wall density:", np.mean(in_arr[0, 0]))
    print("Sample 0 path length:", np.sum(tgt_arr[0, 0]))
