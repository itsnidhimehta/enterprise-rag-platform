import sys
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Make `import rag` work when pytest runs from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def make_pdf(tmp_path):
    """Build a small PDF in memory: one page per string in `pages`."""

    def _make(pages: list[str]) -> bytes:
        path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        for text in pages:
            y = 800
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 15
            c.showPage()
        c.save()
        return path.read_bytes()

    return _make
