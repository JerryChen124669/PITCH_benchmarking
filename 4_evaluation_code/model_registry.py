"""
Reads model_registry.json (model sizes in billions of parameters) and orders
model slugs for plots. Ties on the primary size are broken by the other size,
then by registry order. The anchor model (gemini) is always placed last.

Models present in the data but missing from the registry are NOT silently
placed: order_by_size() raises with the slugs to add, so a newly added model
never lands in an arbitrary spot.
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def order_by_size(slugs, by: str = "total", exclude=None) -> list[str]:
    """by: 'total' or 'active'. Returns slugs small -> large, anchor last."""
    if by not in ("total", "active"):
        raise ValueError("by must be 'total' or 'active'")
    reg = load_registry()
    anchor = reg["anchor_slug"]
    exclude = set(exclude or [])
    entries = {m["slug"]: (i, m) for i, m in enumerate(reg["models"]) if m["slug"]}

    wanted = [s for s in slugs if s not in exclude]
    unknown = [s for s in wanted if s != anchor and s not in entries]
    if unknown:
        raise SystemExit(
            f"ERROR: model(s) not in model_registry.json: {unknown}. "
            "Add their total/active parameter counts there first."
        )

    primary, secondary = ("total_b", "active_b") if by == "total" else ("active_b", "total_b")
    body = sorted(
        (s for s in wanted if s != anchor),
        key=lambda s: (entries[s][1][primary], entries[s][1][secondary], entries[s][0]),
    )
    return body + ([anchor] if anchor in wanted else [])
