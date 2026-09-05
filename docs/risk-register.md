# Risk Register: Inference-Time Scaling Explainer

**Project**: Inference-Time Scaling Explainer  
**Context**: Pathway x DataForge Hackathon  
**Status**: Active Monitoring  

---

| Risk ID | Title | Severity | Likelihood | Impact | Mitigation Strategy | Owner | Trigger / Verification Check |
|---|---|---|---|---|---|---|---|
| **R-001** | Maze task difficulty fails to show clear rising-then-flattening curve | High | High | High | Run a fast hyperparameter/difficulty check in `ml/` before locking training config. Use $8 \times 8$ mazes with variable loop curriculum ($K \in [1, 10]$). Verify accuracy gains are concave before finalizing UI. | ML Engineer / Antigravity | If $K=1$ accuracy > 70% (too easy) or $K=10$ accuracy < 30% (too hard), adjust grid complexity or model channels. |
| **R-002** | User or judge misinterprets slider as live client-side model execution | High | Medium | High | Display explicit badges and text right above the slider: *"Precomputed Model Outputs: Recorded deterministically once per puzzle per loop count"*. Restate clearly in README and UI footer. | Frontend / Antigravity | Audit page text to ensure no deceptive terms like "Model is thinking now..." are present. |
| **R-003** | BDH-CQ frontier results conflated with local toy experiment | High | Medium | High | Visually segregate the BDH-CQ section with distinct styling, clear caption *"Reported by Pathway in arXiv:2608.09888; not independently reproduced for this specific experiment"*, and verbatim Table 5 numbers. | Researcher / Antigravity | Review against Table 5 in `2608.09888v1.md`: LOW 21%, MED 27%, HIGH 29.5%. |
| **R-004** | Training / data generation is non-deterministic or non-reproducible | Medium | Low-Medium | Medium | Fix all random seeds (`torch.manual_seed(42)`, `numpy.random.seed(42)`). Commit both the trained checkpoint `model.pt` and the exported JSON records so the web app runs immediately out of the box, while scripts reproduce identically. | ML Engineer / Antigravity | Re-run `generate_mazes.py` $\rightarrow$ `train.py` $\rightarrow$ `evaluate.py` in `.venv` and verify byte-identical or statistically consistent outputs. |
| **R-005** | Incorrect academic citations or hallucinated paper titles | Medium | Low | High | Enforce verbatim citation from verified paper texts: `arXiv:2509.26507v1` (*"The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"*) and `arXiv:2608.09888v1` (*"BDH-CQ: In-Context Learning with Recurrent Latent Reasoning"*). | Lead / Antigravity | Automated contract test verifies exact paper metadata strings in reference JSON and HTML. |
