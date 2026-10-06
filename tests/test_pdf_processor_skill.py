from __future__ import annotations

from pathlib import Path


SKILL_DIR = Path("agent/skills/pdf-processor")


def test_pdf_processor_skill_bundle_exists():
    skill_md = SKILL_DIR / "SKILL.md"
    inspect_script = SKILL_DIR / "scripts" / "inspect_pdf.py"
    extract_script = SKILL_DIR / "scripts" / "extract_pdf_text.py"
    ocr_script = SKILL_DIR / "scripts" / "ocr_pdf.py"

    assert skill_md.exists()
    assert inspect_script.exists()
    assert extract_script.exists()
    assert ocr_script.exists()

    content = skill_md.read_text(encoding="utf-8")
    assert "name: pdf-processor" in content
    assert "OCR" in content
    assert "ocrmypdf" in content
    assert "page-level" in content
