from __future__ import annotations

from src.cma.enforcement import CMAConfig, CMAEnforcer


def test_cma_enforcement_blocks_without_phases_and_blueprint():
    enforcer = CMAEnforcer(CMAConfig(min_constitutional_score=0.0))
    # Missing required sections and phases
    blueprint = "title: Test\nnotes: just a stub"
    report = enforcer.validate_pre_generation(
        blueprint_text=blueprint, phases_completed=[]
    )
    assert report["ok"] is False
    assert any("Phase missing" in r for r in report["reasons"])


def test_cma_enforcement_accepts_minimal_valid_structure():
    enforcer = CMAEnforcer(CMAConfig(min_constitutional_score=0.0))
    blueprint = """
persona_profile: {name: persona}
narrative_state_machine: {states: [], transitions: []}
sensory_palette: []
thematic_blueprint: {theme: x, message: y}
master_template_library: []
acceptance_criteria: []
"""
    report = enforcer.validate_pre_generation(
        blueprint_text=blueprint,
        phases_completed=["alignment_and_scoping", "blueprint_finalization"],
    )
    assert report["ok"] is True

