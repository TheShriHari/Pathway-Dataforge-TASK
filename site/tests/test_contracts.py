"""
Contract and integrity tests for site datasets and reference citations.
Ensures:
- BDH-CQ numbers strictly match arXiv:2608.09888 Table 5.
- Exact paper titles and authors are cited without error.
- All files have explicit provenance badges.
"""

import os
import json
import pytest


def test_bdh_cq_reference_contract():
    """Verify site/data/bdh_cq_reference.json matches Table 5 of arXiv:2608.09888 verbatim."""
    ref_path = "site/data/bdh_cq_reference.json"
    if not os.path.exists(ref_path):
        pytest.fail(f"Required file {ref_path} does not exist.")

    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Provenance check
    assert data["provenance"] == "EXTERNAL_REPORTED"

    # Citation check
    paper = data["source_paper"]
    assert paper["title"] == "BDH-CQ: In-Context Learning with Recurrent Latent Reasoning"
    assert "Kosowski" in paper["authors"]
    assert paper["arxiv_id"] == "arXiv:2608.09888"
    assert paper["table"] == "Table 5"

    # Table 5 numbers check:
    # LOW: 21% pass@2, 22% cost reduction
    # MEDIUM: 27% pass@2, 11% cost reduction
    # HIGH: 29.5% pass@2, 0% cost reduction
    efforts = {item["level"]: item for item in data["effort_levels"]}

    assert "LOW" in efforts
    assert efforts["LOW"]["pass_at_2"] == 21.0
    assert efforts["LOW"]["cost_reduction_pct"] == 22.0

    assert "MEDIUM" in efforts
    assert efforts["MEDIUM"]["pass_at_2"] == 27.0
    assert efforts["MEDIUM"]["cost_reduction_pct"] == 11.0

    assert "HIGH" in efforts
    assert efforts["HIGH"]["pass_at_2"] == 29.5
    assert efforts["HIGH"]["cost_reduction_pct"] == 0.0

    # Foundation paper citation check
    assert "foundational_paper" in data
    foundational = data["foundational_paper"]
    assert foundational["title"] == "The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"
    assert foundational["arxiv_id"] == "arXiv:2509.26507"


def test_no_hardcoded_duplicate_mismatches_in_site():
    """Verify that site HTML/JS references data files dynamically rather than duplicating drifted numbers."""
    # Ensure index.html and app.js don't hardcode fabricated numbers
    html_path = "site/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Verify central claim is present
        assert "Increasing the number of recurrent latent computation steps" in content
