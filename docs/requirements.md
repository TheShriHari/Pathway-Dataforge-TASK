# Requirements Specification: Inference-Time Scaling Explainer

**Project**: Inference-Time Scaling Explainer  
**Context**: Pathway x DataForge Hackathon  
**Status**: Frozen / Approved  

---

## 1. Central Claim & Purpose

> **Central Claim**:  
> *"Increasing the number of recurrent latent computation steps at inference time can improve reasoning accuracy without producing any additional natural-language reasoning tokens — but gains diminish and each extra step costs more compute/time."*

The purpose of this artifact is to provide an interactive, educational, and scientifically honest explainer demonstrating how recurrent latent reasoning scales compute during inference. It connects a transparent, reproducible toy experiment (maze solving via shared-weight recurrence) with published empirical results from Pathway's frontier research (*BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888, and *The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain*, arXiv:2509.26507).

---

## 2. Target Audience, Prerequisites & Limitations

- **Target Audience**: ML engineers, AI researchers, computer science students, and hackathon judges evaluating inference-time scaling paradigms.
- **Prerequisites**: Basic intuition regarding neural networks, forward passes, and search/pathfinding problems.
- **Explicit Limitations**:
  1. *Task Domain*: The local demonstration solves synthetic 2D grid mazes; while it models recurrent latent computation, grid search dynamics differ from natural language semantics and full ARC-AGI abstractions.
  2. *Scale*: The toy model (~100k parameters) operates on $8 \times 8$ grids, unlike full-scale foundation models (e.g., the 150M parameter BDH-CQ).
  3. *Precomputed Substrate*: To ensure zero-latency, zero-backend, and reproducible client-side exploration, model traces are precomputed across loop counts $K \in [1, 10]$ on a fixed test split rather than executing live client-side inference.

---

## 3. Functional Requirements (FR1 – FR6)

### FR1: Reasoning Effort Slider Control
- The user can view a maze puzzle and interact with a continuous/stepped "Reasoning Effort" slider ranging from $K = 1$ to $K = 10$ recurrent computation loops.
- The control must support both pointer (mouse/touch) dragging and keyboard navigation (Left/Right arrow keys, Home/End, PageUp/PageDown).

### FR2: Side-by-Side Model Prediction & Ground Truth
- When the slider is moved to loop count $K$, the UI updates immediately to render the exact model prediction for that puzzle at step $K$.
- The ground truth solution path is displayed side-by-side (or overlaid with clear visual toggle) for direct qualitative comparison.
- Visual status indicators show whether the model has found a continuous valid path from start to goal at step $K$.

### FR3: Aggregate Accuracy-vs-Effort Evaluation Curve
- The user can inspect an aggregate evaluation chart plotting accuracy (exact maze solve rate and path IoU) versus number of recurrent steps ($K = 1 \dots 10$) across a held-out test set.
- The curve clearly exhibits empirical scaling behavior: increasing accuracy across initial steps followed by diminishing returns (plateauing).
- Interactive hover/focus reveals exact numerical percentages for each step count.

### FR4: Published BDH-CQ Evidence Module
- The artifact presents a dedicated module displaying the developer-reported results from Table 5 of Pathway's BDH-CQ paper (*BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888):
  - **LOW Effort**: 21.0% pass@2, 22% cost reduction vs. Standard
  - **MEDIUM Effort**: 27.0% pass@2, 11% cost reduction vs. Standard
  - **HIGH Effort**: 29.5% pass@2, 0% cost reduction (Standard baseline)
- **Strict Evidence Labeling**: Explicitly states that these numbers are published benchmark results from the BDH-CQ research team on ARC-AGI-1, and are **not** independently reproduced by this toy project.
- Displays full formal citations to both `arXiv:2608.09888` and `arXiv:2509.26507`.

### FR5: Educational Framing & Stated Constraints
- The page prominently features the one-sentence central claim at the top.
- Clear sections for intended audience, conceptual prerequisites, theoretical explanation (how latent recurrence differs from token-generation Chain-of-Thought), and stated limitations are rendered directly on-page.

### FR6: Strict Provenance & Evidence Discipline Labeling
- Every visual component, metric, and data point on the page must carry an unambiguous badge/label:
  - `[PRECOMPUTED REAL]`: Computed from the trained model pipeline on held-out test data.
  - `[EXTERNAL REPORTED]`: Sourced directly from cited academic literature (BDH-CQ Table 5).
  - `[ILLUSTRATIVE]`: Conceptual diagrams or explanatory graphics.
- No ambiguous "live AI thinking" illusions are permitted.

---

## 4. Non-Functional Requirements

### NFR1: Performance & Responsiveness
- Slider interaction and DOM rendering must execute in under 200ms (data is precomputed/static — no live in-browser model inference).
- Initial static bundle load under 500KB (no heavy UI frameworks, no large raster assets).

### NFR2: Accessibility (WCAG 2.1 AA Compliance)
- All interactive controls (slider, puzzle selector, tabs) must be 100% keyboard-navigable with visible focus rings.
- Color contrast ratio must exceed 4.5:1 for normal text and 3.0:1 for graphical UI elements.
- All charts must feature semantic text alternatives (e.g., accessible data table or descriptive `aria-label`/`sr-only` descriptions).

### NFR3: Reproducibility & Determinism
- Fixed random seeds across all data generation, model weight initialization, training curriculum, and test evaluation.
- Running `python ml/generate_mazes.py` $\rightarrow$ `python ml/train.py` $\rightarrow$ `python ml/evaluate.py` must regenerate results matching the committed JSON files.

### NFR4: Constraints
- **Zero Server/Backend**: Fully client-side static web application (HTML5, Vanilla ES6+, CSS3).
- **No Paid APIs or External Cloud Services**: Works 100% offline once loaded.
- **Deployable Anywhere**: Compatible with GitHub Pages, Vercel Static, Cloudflare Pages, or local `file:///` / `python -m http.server`.

---

## 5. Deliverables Tracking & Scope Boundaries

- **Codebase Deliverables (Active Session)**:
  - Python ML pipeline (`ml/`) with deterministic tests and precomputed outputs (`site/data/`).
  - Interactive static web application (`site/`) meeting FR1–FR6 and WCAG AA.
  - Comprehensive `README.md` with methodology, provenance matrix, reproduction commands, and academic citations.
- **Tracked External Deliverable (Outside Coding Session)**:
  - `docs/one-page-summary.pdf`: One-page executive PDF summary for judges, synthesized directly from the final verified artifact and metrics. Tracked as an explicit requirement to ensure it is not dropped.

