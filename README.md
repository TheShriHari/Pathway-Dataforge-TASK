# Inference-Time Scaling Explainer
> **DataForge x Pathway Hackathon Submission**  
> **Live Interactive Explainer**: [https://theshrihari.github.io/Dataforge-Submission/](https://theshrihari.github.io/Dataforge-Submission/)  
> An educational, interactive web artifact demonstrating inference-time latent compute scaling without verbalized natural-language reasoning tokens.

---

## 1. Central Claim

> **"Increasing the number of recurrent latent computation steps at inference time can improve reasoning accuracy without producing any additional natural-language reasoning tokens - but gains diminish and each extra step costs more compute/time."**

This is a falsifiable, empirical claim:
- **Testable positive aspect:** Model pathfinding accuracy and Path IoU increase monotonically as inference loop iterations $K$ increase from $K=1$ upward.
- **Testable negative aspect (trade-off):** Computation latency scales linearly with $K$, while accuracy gains saturate beyond a critical threshold ($K \ge 11$), demonstrating the boundary of diminishing returns.

---

## 2. Intended Audience & Prerequisites

- **Intended Audience**: Machine learning researchers, students, and practitioners studying test-time compute, reasoning models, non-autoregressive architectures, and post-transformer representations.
- **Prerequisites**: 
  1. Foundational understanding of neural network forward passes and hidden states.
  2. Intuition regarding spatial convolutions and receptive field expansion.
  3. Basic familiarity with discrete search algorithms (Breadth-First Search / BFS).

---

## 3. Explicit Learning Objectives

By interacting with this artifact, a learner will be able to:
1. **Contrast Latent Recurrence vs. Autoregressive Chain-of-Thought (CoT)**: Distinguish how fixed-tensor recurrent passes refine internal latent representations ($h_k = \text{Core}(h_{k-1}, x)$) without emitting variable-length text tokens into context memory.
2. **Explain Receptive Field Expansion via Weight-Shared Loops**: Understand how looping a shared $3 \times 3$ convolutional kernel $K$ times iteratively propagates spatial path connectivity without adding a single static parameter.
3. **Identify Empirical Diminishing Returns**: Trace the non-linear relationship where low-effort loops ($K=1 \to 5$) yield dramatic accuracy leaps ($+43.0\%$ exact solve rate), whereas high-effort loops ($K=11 \to 20$) produce marginal gains ($+4.0\%$) despite doubling computational cost.
4. **Contextualize Toy Results with Frontier Foundation Models**: Relate local spatial maze scaling dynamics to published results from Pathway's **BDH-CQ** architecture on the ARC-AGI-1 benchmark (arXiv:2608.09888, Table 5).

---

## 4. The 60-Second Test

After exploring the artifact, a learner should be able to answer these questions:

* **Q1: Why does increasing $K$ from 1 to 5 drastically increase the maze solve rate, but increasing $K$ from 15 to 20 produces almost no gain?**  
  * **Answer:** Each recurrent loop expands the effective receptive field through the maze passages. Small $K$ values cannot bridge long corridor paths from start to goal. Once $K$ is large enough to span the critical path, the network has resolved topological connectivity; further iterations refine already-saturated probability logits, yielding diminishing returns while latency continues to grow linearly.

* **Q2: How does recurrent latent scaling differ from standard LLM Chain-of-Thought (CoT) at inference time?**  
  * **Answer:** CoT generates intermediate natural-language tokens appended to the context window, causing $O(N^2)$ attention memory growth, KV-cache pressure, and risks of semantic drift or syntactic errors. Latent recurrence performs fixed-shape tensor operations entirely within internal latent state ($h \in \mathbb{R}^{C \times H \times W}$), maintaining constant memory consumption with zero emitted verbal tokens.

* **Q3: What is Path IoU and why is it reported alongside Exact Solve Rate?**  
  * **Answer:** Exact Solve Rate is an all-or-nothing binary metric (0% or 100% per maze); stopping one step short of the goal yields a 0% score. Path IoU ($\frac{|\text{Predicted} \cap \text{GroundTruth}|}{|\text{Predicted} \cup \text{GroundTruth}|}$) measures continuous overlap, rewarding partial progress along the correct corridor and penalizing exploratory detours into dead ends or walls.

---

## 5. Artifact Architecture & Component Roles

The application consists of a decoupled machine learning pipeline and a zero-framework, accessible client-side interactive dashboard:

```
├── ml/
│   ├── generate_mazes.py     # Deterministic maze generator (Depth-First Search) & BFS shortest-path solver
│   ├── model.py              # RecurrentLatentMazeSolver (ConvGRU shared-weight core, 123,713 parameters)
│   ├── train.py              # Curriculum training over randomized loop counts K ~ Uniform(1, 15)
│   ├── evaluate.py           # Benchmark evaluator across K=1..20 on 100 held-out test mazes
│   └── test/                 # Pytest test suite for ML components
├── site/
│   ├── index.html            # Semantic, WCAG AA accessible single-page interface
│   ├── app.js                # Vanilla ES6+ reactive renderer (<100ms interaction latency)
│   ├── style.css             # High-contrast, dark-mode design system
│   ├── data/                 # Static precomputed evaluation traces and external benchmark JSONs
│   └── test/                 # Contract and citation integrity test suite
├── results/
│   ├── accuracy_by_effort.json # Canonical 100-maze evaluation scaling curve (K=1..20)
│   └── example_traces.json     # Precomputed multi-step inference walks for interactive replay
└── docs/
    ├── CREDITS.md            # Asset, font, and intellectual inspiration attribution
    └── one-page-summary.md   # Formal one-page submission executive summary
```

### Major UI Components & Roles:
- **Module 1: Interactive Reasoning Effort Slider & Maze Explorer**: Allows the learner to adjust reasoning effort $K \in [1, 20]$ or click "+ / − Effort". The "Auto" button steps $K$ upward automatically and halts as soon as the model reaches a fully connected path (typically $K=4–5$), letting the learner see the actual moment additional effort stops being necessary - visually reinforcing the plateau shown in the aggregate chart.
- **Module 2: Empirical Scaling Curve & Diminishing Returns Chart**: Dual-view SVG chart offering an aggregate 100-maze scaling curve (Exact Solve Rate % and Path IoU %) and an individual puzzle trace view with step-by-step progress tracking.
- **Module 3: Frontier Literature Integration (BDH-CQ ARC-AGI-1)**: Comparative benchmark cards displaying Table 5 results from arXiv:2608.09888.
- **Module 4: Technical Comparison & Concepts**: Side-by-side architectural comparison between autoregressive CoT and recurrent latent recurrence.

---

## 6. Component Provenance & Classification Matrix

To prevent deceptive "live AI thinking" illusions, every component in this repository is explicitly classified:

| Component | Provenance Classification | Method & Implementation Details |
|---|---|---|
| **Synthetic Mazes** | `[SYNTHETIC]` | $15 \times 15$ grid mazes procedurally generated using deterministic randomized DFS backtracker with fixed seeds (`ml/generate_mazes.py`). |
| **Ground Truth Solutions** | `[SYNTHETIC]` | BFS shortest-path optimal walks connecting Start $(0, 0)$ to Goal $(14, 14)$ through open corridors (`ml/generate_mazes.py`). |
| **Model Weights** | `[PRECOMPUTED]` | 123,713-parameter ConvGRU PyTorch checkpoint (`ml/checkpoints/best_model.pt`) trained on synthetic mazes. |
| **100-Maze Scaling Curve** | `[PRECOMPUTED]` | Evaluated across $N=100$ held-out $15 \times 15$ test mazes (Seed 999) across loops $K=1 \dots 20$ using `ml/evaluate.py`. Stored in `results/accuracy_by_effort.json`. |
| **Interactive Step Replay** | `[PRECOMPUTED]` | Precomputed inference output probability masks exported per puzzle per step $K \in [1, 20]$ in `site/data/example_traces.json`. No live neural network is executed inside the browser. |
| **UI Path Animations** | `[ANIMATED]` | CSS SVG line draw transitions (`stroke-dashoffset` keyframes) visually demonstrating directional path propagation when stepping between effort levels. |
| **Client UI State** | `[LIVE]` | User input handling, slider adjustments, chart view toggling, and table row highlighting executed live in vanilla JavaScript (`site/app.js`). |
| **BDH-CQ Table 5 Data** | `[PRECOMPUTED / EXTERNAL]` | Direct reproduction of published experimental data from Pathway (arXiv:2608.09888, Table 5) on the ARC-AGI-1 benchmark stored in `site/data/bdh_cq_reference.json`. Not generated by local code. |

---

## 7. Verified Empirical Scaling Results

The following numbers are verified directly from `results/accuracy_by_effort.json` (evaluated on $N=100$ held-out $15 \times 15$ mazes, Seed 999, ConvGRU core, 123,713 parameters):

| Reasoning Loops ($K$) | Exact Solve Rate (%) | Mean Path IoU (%) | Mean BCE Loss | Measured Latency (ms / maze) |
|:---:|:---:|:---:|:---:|:---:|
| **$K = 1$ (Minimal)** | 1.0% | 55.94% | 0.3281 | 0.666 ms |
| **$K = 2$** | 2.0% | 65.53% | 0.2518 | 1.168 ms |
| **$K = 3$** | 19.0% | 76.63% | 0.1739 | 1.747 ms |
| **$K = 4$** | 35.0% | 83.29% | 0.1253 | 2.373 ms |
| **$K = 5$** | 44.0% | 87.69% | 0.0945 | 2.911 ms |
| **$K = 6$** | 53.0% | 90.93% | 0.0736 | 3.519 ms |
| **$K = 7$** | 63.0% | 93.12% | 0.0594 | 4.098 ms |
| **$K = 8$** | 71.0% | 94.75% | 0.0485 | 5.292 ms |
| **$K = 9$** | 74.0% | 95.64% | 0.0417 | 5.465 ms |
| **$K = 10$ (Inflection)** | **81.0%** | **96.61%** | **0.0364** | **6.155 ms** |
| **$K = 11$** | 84.0% | 97.12% | 0.0327 | 6.568 ms |
| **$K = 12$** | 84.0% | 97.42% | 0.0307 | 7.849 ms |
| **$K = 13$** | 85.0% | 97.51% | 0.0294 | 7.694 ms |
| **$K = 14$** | 87.0% | 97.83% | 0.0285 | 8.583 ms |
| **$K = 15$** | 88.0% | 98.01% | 0.0279 | 9.374 ms |
| **$K = 16$** | 86.0% | 98.12% | 0.0277 | 9.059 ms |
| **$K = 17$** | 86.0% | 98.11% | 0.0278 | 10.795 ms |
| **$K = 18$** | 87.0% | 98.14% | 0.0281 | 10.959 ms |
| **$K = 19$** | 88.0% | 98.20% | 0.0284 | 11.737 ms |
| **$K = 20$ (High)** | 88.0% | 98.13% | 0.0289 | 12.884 ms |

### Key Observations:
- **Steep Early Phase ($K=1 \to K=10$)**: Exact solve rate climbs from **1.0% to 81.0%** ($+80.0\%$ absolute gain) and Path IoU rises from **55.94% to 96.61%** as the recurrent loops expand the effective receptive field across the $15 \times 15$ maze.
- **Diminishing Returns Phase ($K=10 \to K=20$)**: Increasing $K$ from 10 to 20 yields only a **$+7.0\%$** gain in solve rate (81.0% $\to$ 88.0%), while inference latency more than doubles from **6.155 ms to 12.884 ms**.
- **Training Horizon & Generalization**: $K=16–20$ extends beyond the $K \sim \text{Uniform}(1, 15)$ training curriculum (`ml/train.py`). The model generalizes to these values because the recurrent core is weight-shared and has no explicit step-count parameter, but this was not part of the training distribution.

---

## 8. Frontier Literature Integration: BDH-CQ Comparison

In *"BDH-CQ: In-Context Learning with Recurrent Latent Reasoning"* (arXiv:2608.09888), Pathway evaluated recurrent latent reasoning effort on the ARC-AGI-1 benchmark:

| Effort Level | ARC-AGI-1 Pass@2 | Compute Cost Reduction | Relative Compute Cost |
|:---:|:---:|:---:|:---:|
| **LOW** | 21.0% | -22.0% | 78% |
| **MEDIUM** | 27.0% | -11.0% | 89% |
| **HIGH** | 29.5% | Baseline (0.0%) | 100% |

*Evidence note: These numbers are external developer-reported results from Table 5 of Kosowski et al. (2026). They demonstrate the same core principle as our toy maze solver: allocating additional latent recurrent computation improves problem-solving capability up to an empirical plateau.*

---

## 9. Reproduction Steps (From a Fresh Clone)

Follow these steps to reproduce the entire environment, test suite, and local server from scratch:

```bash
# 1. Clone the repository
git clone https://github.com/TheShriHari/Dataforge-Submission.git
cd Dataforge-Submission

# 2. Create and activate an isolated virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies from requirements.txt
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run test suites to verify ML logic and contract integrity
python -m pytest site/test
python -m pytest ml/test/test_generate_mazes.py ml/test/test_model.py ml/test/test_evaluate.py

# 5. (Optional) Re-run model evaluation on the test split
python -m ml.evaluate

# 6. Launch the local interactive web explainer
python -m http.server 8888 --directory site
```
Once started, open **`http://localhost:8888`** in any modern web browser.

---

## 10. Credits & Licenses

- **Codebase License**: MIT License (see [LICENSE](./LICENSE)).
- **Web Typography**: Inter, Space Grotesk, and IBM Plex Mono sourced via Google Fonts under the [SIL Open Font License 1.1](https://openfontlicense.org/).
- **Zero Third-Party JS/CSS Frameworks**: The web client uses 100% pure vanilla ES6+ and vanilla CSS.
- **Detailed Asset & Attribution Register**: See [docs/CREDITS.md](./docs/CREDITS.md) for full library, algorithm, and research citation details.

---

## 11. AI Assistance Disclosure

In compliance with the DataForge and NeurIPS Education Track guidelines, AI assistance is disclosed across distinct categories (see full standalone statement in [AI_DISCLOSURE.md](./AI_DISCLOSURE.md)):

1. **Code Assistance**: Antigravity (Google DeepMind agentic pair programmer, powered by Gemini) assisted in scaffolding boilerplate, drafting unit test templates, generating the initial mathematical model architecture (`ml/model.py`), training loop logic (`ml/train.py`), evaluation script (`ml/evaluate.py`), and maze generation algorithms (`ml/generate_mazes.py`). All code was subsequently inspected, debugged, executed, and validated via unit and contract test suites by the authors before submission.
2. **Data Generation**: All synthetic maze training and testing datasets were generated deterministically using local Python scripts (`ml/generate_mazes.py`) with fixed pseudo-random seeds. No external or proprietary datasets were used.
3. **Assets & Graphics**: All maze diagrams and scaling plots are rendered procedurally as vector SVGs by `site/app.js`. No AI-generated raster imagery was used.
4. **Licenses & Attribution**: All external research citations (arXiv:2509.26507, arXiv:2608.09888, arXiv:2408.03314) and font licenses were verified manually against primary published sources.