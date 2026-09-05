# Implementation Tasks: Inference-Time Scaling Explainer

**Project**: Inference-Time Scaling Explainer  
**Context**: Pathway x DataForge Hackathon  
**Status**: Active  

---

## Task Breakdown & Roadmap

### Phase 1: Test-Driven Development (TDD) Harness
*Objective: Create rigorous automated test suites before any implementation code is written.*
- [ ] **Task 1.1: Maze Generator Unit Tests (`ml/tests/test_generate_mazes.py`)**
  - Verify generated mazes have valid grid dimensions ($8 \times 8$).
  - Verify every generated maze is guaranteed solvable via BFS (shortest path exists from $(0,0)$ to $(7,7)$).
  - Verify wall and path tensors have correct shapes `(3, 8, 8)` and `(1, 8, 8)`.
  - Verify deterministic reproducibility with identical random seeds.
- [ ] **Task 1.2: Recurrent Model Unit Tests (`ml/tests/test_model.py`)**
  - Verify model accepts variable loop count $K$ as input parameter.
  - Verify output tensor shape is `(batch, 1, 8, 8)` for any valid $K \ge 1$.
  - Verify recurrent execution: output logits for $K=1$ must differ from $K=10$ (confirms shared-weight iterations actively alter state).
  - **Fixed Hand-Verifiable Toy Maze Test**: On a fixed deterministic mini-maze with trained or initialized weights, assert that $K=10$ output is strictly closer to ground truth (lower MSE/BCE or higher IoU) than $K=1$, proving that recurrent steps improve solution quality.
  - Verify parameter count is within budget (< 200k params).
- [ ] **Task 1.3: Evaluation & Monotonicity Tests (`ml/tests/test_evaluate.py`)**
  - Verify `evaluate.py` outputs valid schema JSON files (`accuracy_by_effort.json`, `example_traces.json`).
  - Assert that High-K ($K=10$) accuracy meets or exceeds Low-K ($K=1$) accuracy on held-out test data within tolerance ($+0.20$ IoU margin).
  - Assert that the rate of gain shrinks across steps (diminishing returns / concave curve: $(A_{K=5} - A_{K=1}) > (A_{K=10} - A_{K=5})$).
- [ ] **Task 1.4: Site Data Contract & DOM Verification Tests (`site/tests/test_contracts.py`)**
  - Validate JSON schemas for `accuracy_by_effort.json`, `example_traces.json`, and `bdh_cq_reference.json`.
  - Assert Table 5 values in `bdh_cq_reference.json` match arXiv:2608.09888 exactly (21.0%, 27.0%, 29.5%, cost reductions 22%, 11%, 0%).

---

### Phase 2: ML Pipeline Implementation
*Objective: Implement generator, model, training curriculum, and export evaluation traces.*
- [ ] **Task 2.0: Maze Difficulty & Size Sweep Spike (`ml/sweep_difficulty.py`)**
  - Test at least two maze configurations (e.g., $6 \times 6$ vs $8 \times 8$, or dense vs sparse walls).
  - Measure baseline accuracy at $K=1$ vs $K=10$.
  - Confirm which configuration yields the clearest rising-then-plateauing curve (avoiding $K=1$ ceiling >70% or floor <30%).
  - Report findings and lock the chosen configuration into `train.py`.
- [ ] **Task 2.1: Maze Generator (`ml/generate_mazes.py`)**
  - Implement recursive backtracking / Kruskal algorithm with fixed seed.
  - Implement BFS path solver to compute optimal shortest path mask.
  - Export test split and training split tensors.
  - Run `pytest ml/tests/test_generate_mazes.py` to confirm green.
- [ ] **Task 2.2: Recurrent Neural Network Core (`ml/model.py`)**
  - Implement input embedding Conv2D layer.
  - Implement shared-weight recurrent residual block (Conv2D + LayerNorm + GELU).
  - Implement readout projection head with sigmoid activation.
  - Run `pytest ml/tests/test_model.py` to confirm green.
- [ ] **Task 2.3: Variable-Loop Curriculum Training (`ml/train.py`)**
  - Implement training loop with random loop count sampling $K \sim \text{Uniform}(1, 10)$ per batch.
  - Implement combined BCE + Dice loss.
  - **Concrete Convergence Criteria**:
    - Final epoch training BCE loss $< 0.15$.
    - Validation path IoU at $K=10$ exceeds $0.70$.
    - Validation path IoU at $K=10$ exceeds $K=1$ by at least $+0.20$.
  - Train model for 30-50 epochs (fast CPU execution < 2 minutes).
  - Save best checkpoint to `ml/checkpoints/best_model.pt`.
- [ ] **Task 2.4: Evaluation & Trace Export (`ml/evaluate.py`)**
  - Evaluate checkpoint on 100 held-out test mazes for $K=1 \dots 10$.
  - Compute exact solve rates, path IoU, and cross-entropy loss.
  - Export `site/data/accuracy_by_effort.json` and `site/data/example_traces.json`.
  - Create `site/data/bdh_cq_reference.json` with verbatim Table 5 citations.
  - Run `pytest ml/tests/test_evaluate.py` to confirm green.

---

### Phase 3: Client-Side Web Artifact Implementation
*Objective: Build an interactive, accessible static site driven by the precomputed JSON.*
- [ ] **Task 3.1: HTML Structure & Accessibility Scaffolding (`site/index.html`)**
  - Semantic HTML5 structure (header, main, section, nav, footer).
  - Hero section with the one-sentence central claim.
  - Maze Explorer section with slider ($K=1 \dots 10$), puzzle tabs, side-by-side display.
  - Aggregate Scaling Curve container.
  - BDH-CQ Table 5 Benchmark comparison container.
  - Transparency & Provenance table (`[PRECOMPUTED REAL]`, `[EXTERNAL REPORTED]`, `[ILLUSTRATIVE]`).
  - Academic citations and limitations section.
- [ ] **Task 3.2: Modern Accessible Styling (`site/style.css`)**
  - Responsive layout (CSS Grid / Flexbox).
  - Dark-mode technical design system with high-contrast color tokens (WCAG AA).
  - Focus rings, keyboard indicators, step tick marks for the slider.
  - Responsive SVG chart styling and cell heatmap gradients.
- [ ] **Task 3.3: Interactive Logic & Rendering (`site/app.js`)**
  - Asynchronous loading of `accuracy_by_effort.json`, `example_traces.json`, `bdh_cq_reference.json`.
  - Slider event listener updating maze prediction, heatmap, path overlay, and metrics in < 10ms.
  - Accessible SVG rendering for the accuracy scaling curve with interactive tooltips.
  - Side-by-side BDH-CQ external benchmark comparison chart.
  - Screen-reader accessible summary tables.

---

### Phase 4: Full System Verification & Quality Gates
*Objective: Verify all acceptance criteria, functional requirements, and accessibility.*
- [ ] **Task 4.1: Automated Test Suite Execution**
  - Run complete `pytest` suite on all ML tests and contract validators.
- [ ] **Task 4.2: Browser Testing & Accessibility Audit**
  - Validate keyboard navigation (Tab order, Arrow keys on slider).
  - Verify contrast ratios pass WCAG AA.
  - Verify screen-reader text alternatives for all visual charts.
- [ ] **Task 4.3: Acceptance Criteria Mapping**
  - Verify FR1-FR6 directly against UI behavior.
  - Confirm all numbers match JSON outputs without fabrication.

---

### Phase 5: Documentation & Reproducibility
*Objective: Complete comprehensive, honest documentation.*
- [ ] **Task 5.1: Master README.md**
  - Central claim, target audience, prerequisites, limitations.
  - Provenance matrix (what is live vs. precomputed vs. illustrative).
  - Step-by-step reproduction commands using `.venv`.
  - Academic citations with correct titles and arXiv links.
  - AI assistance disclosure.
- [ ] **Task 5.2: Final Checkpoint Summary**
  - Log completed deliverables, metrics, and outline for one-page PDF summary.

---

### Phase 6: External Deliverables Tracking (Judges Package)
*Objective: Track deliverables created after the working web artifact is verified.*
- [ ] **Task 6.1: One-Page Executive Summary PDF (`docs/one-page-summary.pdf`)**
  - Formatted one-page deliverable for hackathon judges summarizing the central claim, empirical scaling results, BDH-CQ contextualization, and live artifact URL.
- [ ] **Task 6.2: Academic Citations & Paper Verification Registry**
  - Formal BibTeX and markdown references for arXiv:2509.26507 and arXiv:2608.09888.
