from __future__ import annotations

from pathlib import Path


SKILL_DIR = Path("agent/skills/wage-dispute-workflow")


def test_wage_dispute_skill_bundle_exists():
    skill_md = SKILL_DIR / "SKILL.md"

    assert skill_md.exists()

    content = skill_md.read_text(encoding="utf-8")
    assert "name: wage-dispute-workflow" in content
    assert "工资争议咨询流程" in content
    assert "claim item" in content
    assert "劳动监察" in content
    assert "Do not cite statutes" in content
