from __future__ import annotations

from pathlib import Path


SKILL_DIR = Path("agent/skills/legal-pleading-drafter")


def test_legal_pleading_drafter_skill_bundle_exists():
    skill_md = SKILL_DIR / "SKILL.md"
    document_types = SKILL_DIR / "references" / "document-types.md"
    fact_checklist = SKILL_DIR / "references" / "fact-checklist.md"
    pleading_style = SKILL_DIR / "references" / "pleading-style.md"
    quality_review = SKILL_DIR / "references" / "quality-review.md"
    punctuation_script = SKILL_DIR / "scripts" / "fix_punctuation.py"

    assert skill_md.exists()
    assert document_types.exists()
    assert fact_checklist.exists()
    assert pleading_style.exists()
    assert quality_review.exists()
    assert punctuation_script.exists()

    content = skill_md.read_text(encoding="utf-8")
    assert "name: legal-pleading-drafter" in content
    assert "起诉状" in content
    assert "答辩状" in content
    assert "去 AI 腔" in content
    assert "不得虚构" in content
