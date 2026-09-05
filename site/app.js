/**
 * Inference-Time Scaling Explainer: Interactive Application Logic (site/app.js)
 * Zero external frameworks. Pure vanilla ES6+ with accessible SVG charting,
 * keyboard controls, and real precomputed trace replay.
 */

// Application State
const state = {
  accuracyData: null,
  tracesData: null,
  bdhData: null,
  currentPuzzleIdx: 0,
  currentK: 1,
};

// Fallback embedded data in case file:/// protocol blocks local fetch
const DEFAULT_BDH_DATA = {
  provenance: "EXTERNAL_REPORTED",
  source_paper: {
    title: "BDH-CQ: In-Context Learning with Recurrent Latent Reasoning",
    authors: "Kosowski et al.",
    arxiv_id: "arXiv:2608.09888",
    table: "Table 5"
  },
  effort_levels: [
    { level: "LOW", pass_at_2: 21.0, cost_reduction_pct: 22.0, relative_cost_pct: 78.0, desc: "Reduced reasoning effort: -22% compute cost, 21.0% pass@2." },
    { level: "MEDIUM", pass_at_2: 27.0, cost_reduction_pct: 11.0, relative_cost_pct: 89.0, desc: "Intermediate reasoning effort: +6.0% accuracy gain, -11% compute cost." },
    { level: "HIGH", pass_at_2: 29.5, cost_reduction_pct: 0.0, relative_cost_pct: 100.0, desc: "Full reasoning effort: 29.5% pass@2 baseline with diminishing returns." }
  ]
};

/**
 * Initialize application on DOM load
 */
document.addEventListener("DOMContentLoaded", async () => {
  await loadData();
  setupEventListeners();
  renderPuzzleTabs();
  updateMazeDisplay();
  renderScalingChart();
  renderBdhCards();
});

/**
 * Loads precomputed JSON data records
 */
async function loadData() {
  try {
    const [accRes, trcRes, bdhRes] = await Promise.all([
      fetch("data/accuracy_by_effort.json").catch(() => fetch("../results/accuracy_by_effort.json")),
      fetch("data/example_traces.json").catch(() => fetch("../results/example_traces.json")),
      fetch("data/bdh_cq_reference.json").catch(() => fetch("../results/bdh_cq_reference.json")),
    ]);

    if (accRes && accRes.ok) state.accuracyData = await accRes.json();
    if (trcRes && trcRes.ok) state.tracesData = await trcRes.json();
    if (bdhRes && bdhRes.ok) state.bdhData = await bdhRes.json();
  } catch (err) {
    console.warn("Local fetch restricted (likely file:// protocol), using loaded fallback context:", err);
  }

  // Ensure BDH data is available
  if (!state.bdhData) {
    state.bdhData = DEFAULT_BDH_DATA;
  }
}

/**
 * Attach UI event listeners
 */
function setupEventListeners() {
  const slider = document.getElementById("effort-slider");
  const stepDecrement = document.getElementById("btn-step-prev");
  const stepIncrement = document.getElementById("btn-step-next");

  if (slider) {
    slider.addEventListener("input", (e) => {
      state.currentK = parseInt(e.target.value, 10);
      updateMazeDisplay();
    });
  }

  if (stepDecrement) {
    stepDecrement.addEventListener("click", () => {
      if (state.currentK > 1) {
        state.currentK--;
        if (slider) slider.value = state.currentK;
        updateMazeDisplay();
      }
    });
  }

  if (stepIncrement) {
    stepIncrement.addEventListener("click", () => {
      if (state.currentK < 10) {
        state.currentK++;
        if (slider) slider.value = state.currentK;
        updateMazeDisplay();
      }
    });
  }

  // Keyboard navigation shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type === "range") return;
    if (e.key === "ArrowLeft" && state.currentK > 1) {
      state.currentK--;
      if (slider) slider.value = state.currentK;
      updateMazeDisplay();
    } else if (e.key === "ArrowRight" && state.currentK < 10) {
      state.currentK++;
      if (slider) slider.value = state.currentK;
      updateMazeDisplay();
    }
  });
}

/**
 * Render puzzle selection tabs
 */
function renderPuzzleTabs() {
  const tabsContainer = document.getElementById("puzzle-tabs");
  if (!tabsContainer || !state.tracesData || !state.tracesData.puzzles) return;

  tabsContainer.innerHTML = "";
  state.tracesData.puzzles.forEach((puzzle, idx) => {
    const btn = document.createElement("button");
    btn.className = `tab-btn ${idx === state.currentPuzzleIdx ? "active" : ""}`;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", idx === state.currentPuzzleIdx ? "true" : "false");
    btn.textContent = puzzle.title || `Puzzle ${idx + 1}`;
    btn.addEventListener("click", () => {
      state.currentPuzzleIdx = idx;
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      updateMazeDisplay();
    });
    tabsContainer.appendChild(btn);
  });
}

/**
 * Update maze grids and live metrics based on selected puzzle and K
 */
function updateMazeDisplay() {
  const effortNumEl = document.getElementById("effort-val-display");
  if (effortNumEl) effortNumEl.textContent = state.currentK;

  if (!state.tracesData || !state.tracesData.puzzles) return;

  const puzzle = state.tracesData.puzzles[state.currentPuzzleIdx];
  if (!puzzle) return;

  const kStr = String(state.currentK);
  const stepData = puzzle.steps[kStr];

  // Update Status Pill
  const statusContainer = document.getElementById("maze-status-pill");
  if (statusContainer && stepData) {
    if (stepData.is_solved) {
      statusContainer.className = "status-pill status-solved";
      statusContainer.innerHTML = "<span>&#10003;</span> Path Solved &amp; Connected";
    } else {
      statusContainer.className = "status-pill status-unsolved";
      statusContainer.innerHTML = "<span>&#9888;</span> Incomplete / Dead End";
    }
  }

  // Update Live Metrics
  const metricIou = document.getElementById("metric-iou");
  const metricSolve = document.getElementById("metric-solve");
  const metricCompute = document.getElementById("metric-compute");

  if (metricIou && stepData) metricIou.textContent = `${(stepData.iou * 100).toFixed(1)}%`;
  if (metricSolve && stepData) metricSolve.textContent = stepData.is_solved ? "YES" : "NO";
  if (metricCompute) metricCompute.textContent = `K = ${state.currentK}`;

  // Render Predicted Grid
  const predBoard = document.getElementById("pred-maze-board");
  if (predBoard) {
    renderGrid(predBoard, puzzle, stepData ? stepData.predicted_path : [], false);
  }

  // Render Ground Truth Grid
  const gtBoard = document.getElementById("gt-maze-board");
  if (gtBoard) {
    renderGrid(gtBoard, puzzle, puzzle.ground_truth_path, true);
  }
}

/**
 * Render an 8x8 maze grid
 */
function renderGrid(container, puzzle, pathCoords, isGroundTruth) {
  container.innerHTML = "";
  const size = puzzle.grid_size || 8;
  const pathSet = new Set(pathCoords.map(([r, c]) => `${r},${c}`));

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell = document.createElement("div");
      cell.className = "maze-cell";
      const isWall = puzzle.walls[r][c] === 1;
      const isStart = puzzle.start[0] === r && puzzle.start[1] === c;
      const isGoal = puzzle.goal[0] === r && puzzle.goal[1] === c;
      const isPath = pathSet.has(`${r},${c}`);

      if (isWall) {
        cell.classList.add("wall");
      } else {
        cell.classList.add("passage");
      }

      if (isStart) cell.classList.add("start");
      if (isGoal) cell.classList.add("goal");

      if (isPath && !isStart && !isGoal) {
        cell.classList.add(isGroundTruth ? "gt-path" : "pred-path");
      }

      container.appendChild(cell);
    }
  }
}

/**
 * Render the Aggregate Accuracy vs. Effort SVG Line Chart (FR3)
 */
function renderScalingChart() {
  const chartContainer = document.getElementById("scaling-chart-wrap");
  if (!chartContainer || !state.accuracyData || !state.accuracyData.scaling_curve) return;

  const curve = state.accuracyData.scaling_curve;
  const width = 800;
  const height = 260;
  const padLeft = 60;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  // Scale functions
  const xPos = (k) => padLeft + ((k - 1) / 9) * chartW;
  const yPos = (pct) => padTop + chartH - (pct / 100) * chartH;

  // Build SVG
  let svg = `<svg viewBox="0 0 ${width} ${height}" class="svg-chart" role="img" aria-label="Line chart showing exact solve rate and path IoU increasing and plateauing across reasoning steps 1 to 10">`;

  // Grid lines
  for (let yPct = 0; yPct <= 100; yPct += 20) {
    const y = yPos(yPct);
    svg += `<line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="#1e293b" stroke-dasharray="3 3"/>`;
    svg += `<text x="${padLeft - 10}" y="${y + 4}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">${yPct}%</text>`;
  }

  // X Axis Ticks
  for (let k = 1; k <= 10; k++) {
    const x = xPos(k);
    svg += `<line x1="${x}" y1="${padTop + chartH}" x2="${x}" y2="${padTop + chartH + 5}" stroke="#475569"/>`;
    svg += `<text x="${x}" y="${padTop + chartH + 20}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="monospace">K=${k}</text>`;
  }

  // Draw Exact Solve Rate Line (Cyan/Emerald)
  const solvePoints = curve.map((pt) => `${xPos(pt.step)},${yPos(pt.exact_solve_percent)}`).join(" ");
  svg += `<polyline fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${solvePoints}"/>`;

  // Draw Path IoU Line (Violet)
  const iouPoints = curve.map((pt) => `${xPos(pt.step)},${yPos(pt.path_iou * 100)}`).join(" ");
  svg += `<polyline fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-dasharray="4 4" stroke-linecap="round" points="${iouPoints}"/>`;

  // Data Dots
  curve.forEach((pt) => {
    const x = xPos(pt.step);
    const ySolve = yPos(pt.exact_solve_percent);
    svg += `<circle cx="${x}" cy="${ySolve}" r="5" fill="#10b981" stroke="#0a0d14" stroke-width="2"><title>Step K=${pt.step}: Exact Solve = ${pt.exact_solve_percent}%</title></circle>`;
  });

  svg += `</svg>`;
  chartContainer.innerHTML = svg;

  // Populate accessible table
  const tbody = document.getElementById("table-scaling-body");
  if (tbody) {
    tbody.innerHTML = "";
    curve.forEach((pt) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>K = ${pt.step}</strong></td>
        <td style="color: #34d399; font-weight: 700;">${pt.exact_solve_percent}%</td>
        <td style="color: #c084fc;">${(pt.path_iou * 100).toFixed(1)}%</td>
        <td>${pt.mean_bce_loss.toFixed(4)}</td>
        <td style="font-family: monospace; color: #38bdf8;">${pt.measured_latency_ms.toFixed(3)} ms</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

/**
 * Render BDH-CQ Table 5 Benchmark Cards (FR4)
 */
function renderBdhCards() {
  const container = document.getElementById("bdh-cards-container");
  if (!container || !state.bdhData || !state.bdhData.effort_levels) return;

  container.innerHTML = "";
  state.bdhData.effort_levels.forEach((item) => {
    const card = document.createElement("div");
    card.className = `bdh-card ${item.level === "HIGH" ? "highlight" : ""}`;
    card.innerHTML = `
      <div class="bdh-card-level">${item.level} EFFORT</div>
      <div class="bdh-card-metric">${item.pass_at_2.toFixed(1)}%</div>
      <div class="bdh-card-cost">Compute Cost: ${item.cost_reduction_pct > 0 ? `-${item.cost_reduction_pct.toFixed(0)}%` : "Baseline (100%)"}</div>
      <p class="bdh-card-desc">${item.description || item.desc || ""}</p>
    `;
    container.appendChild(card);
  });
}
