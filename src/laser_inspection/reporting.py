"""JSON report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .models import InspectionResult


def save_report(path: str | Path, result: InspectionResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
