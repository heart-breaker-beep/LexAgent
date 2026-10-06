from __future__ import annotations

from pathlib import Path


SKILL_DIR = Path("agent/skills/video-screenshot")


def test_video_screenshot_skill_bundle_exists():
    skill_md = SKILL_DIR / "SKILL.md"
    extract_script = SKILL_DIR / "scripts" / "extract.py"
    lib_script = SKILL_DIR / "scripts" / "lib.py"
    setup_ref = SKILL_DIR / "references" / "setup.md"
    strategy_ref = SKILL_DIR / "references" / "strategy-and-params.md"

    assert skill_md.exists()
    assert extract_script.exists()
    assert lib_script.exists()
    assert setup_ref.exists()
    assert strategy_ref.exists()

    content = skill_md.read_text(encoding="utf-8")
    assert "name: video-screenshot" in content
    assert "微信聊天录屏" in content
    assert "SHA256" in content
    assert "_report.json" in content
