"""Laya-backed vulnerability deduplication decisions.

Replaces the LLM dedupe judge with local decision-model passes when Laya is
enabled: each pass asks one choice question over the ids of a chunk of
existing reports ("none" plus up to ``_REPORTS_PER_PASS`` ids) and answers it
in a single forward pass with a calibrated confidence.

Every path here returns ``None`` — meaning "escalate to the LLM judge" —
whenever Laya is disabled or not installed, the state does not fit the
checkpoint context, the answer is below ``STRIX_LAYA_MIN_CONFIDENCE``, or
anything raises. Laya reduces LLM dependency; it never blocks a decision on
itself.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from strix.config import load_settings
from strix.laya.engine import get_engine


if TYPE_CHECKING:
    from strix.laya.engine import LayaEngine

logger = logging.getLogger(__name__)

# laya.structured caps one choice question at 32 options; "none" takes one
# slot, so a chunk never carries more than 31 existing reports. The default is
# far lower to keep the serialized state inside the checkpoint's token context
# (typed-decisions has 1024).
_MAX_OPTIONS = 32
_REPORTS_PER_PASS = 8
# Per-field truncation for the serialized state; digests carry only the fields
# the LLM judge rules on, so a short window is enough.
_FIELD_CHARS = 300

_DIGEST_FIELDS = (
    "id",
    "title",
    "description",
    "impact",
    "target",
    "endpoint",
    "method",
    "technical_analysis",
    "poc_description",
    "cve",
)

DEDUPE_DECISION_INSTRUCTIONS = (
    "Which existing report describes the SAME vulnerability as the candidate? "
    "Same vulnerability means same root cause, same affected endpoint/parameter "
    "or file, and the same fix. Different endpoints, parameters, root causes, or "
    "auth requirements are NOT duplicates even when the vulnerability type "
    "matches. Answer 'none' when the candidate is a distinct finding."
)


async def check_duplicate_with_laya(
    candidate: dict[str, Any],
    existing_reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Decide the candidate against ``existing_reports`` with Laya.

    Returns the dedupe result dict when Laya answered confidently, or ``None``
    to escalate the decision to the LLM judge.
    """
    laya = load_settings().laya
    if not laya.enabled or not laya.dedupe:
        return None
    engine = get_engine()
    if engine is None:
        return None
    return await _decide_all(engine, laya.min_confidence, candidate, existing_reports)


async def _decide_all(
    engine: LayaEngine,
    min_confidence: float,
    candidate: dict[str, Any],
    existing_reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve the candidate against every chunk; None means escalate."""
    best_duplicate: dict[str, Any] | None = None
    lowest_confidence = 1.0
    try:
        for offset in range(0, len(existing_reports), _REPORTS_PER_PASS):
            chunk = existing_reports[offset : offset + _REPORTS_PER_PASS]
            answer = await _decide_chunk(engine, candidate, chunk)
            if answer is None:
                return None
            value, confidence = answer
            if confidence < min_confidence:
                return None
            lowest_confidence = min(lowest_confidence, confidence)
            if value == "none":
                continue
            if best_duplicate is None or confidence > best_duplicate["confidence"]:
                best_duplicate = {
                    "is_duplicate": True,
                    "duplicate_id": value[:64],
                    "confidence": confidence,
                    "reason": (
                        "Laya decision model: same vulnerability as existing report "
                        f"{value} (confidence {confidence:.2f})"
                    ),
                }
    except Exception:
        logger.exception("Laya dedupe decision failed; escalating to the LLM")
        return None
    if best_duplicate is not None:
        return best_duplicate
    return {
        "is_duplicate": False,
        "duplicate_id": "",
        "confidence": lowest_confidence,
        "reason": (
            f"Laya decision model: no existing report matches (confidence {lowest_confidence:.2f})"
        ),
    }


async def _decide_chunk(
    engine: LayaEngine,
    candidate: dict[str, Any],
    chunk: list[dict[str, Any]],
) -> tuple[str, float] | None:
    """One forward pass over one chunk; None means escalate."""
    ids = [str(report.get("id") or "") for report in chunk]
    if any(not report_id for report_id in ids) or len(ids) > _MAX_OPTIONS - 1:
        return None
    schema = {
        "type": "object",
        "properties": {
            "duplicate_of": {
                "type": "string",
                "enum": ["none", *ids],
                "description": DEDUPE_DECISION_INSTRUCTIONS,
            }
        },
        "required": ["duplicate_of"],
    }
    state = json.dumps(
        {"candidate": _digest(candidate), "existing_reports": [_digest(r) for r in chunk]},
        ensure_ascii=False,
        default=str,
    )
    result = await engine.decide(state, schema)
    if result is None:
        return None
    value = result["values"].get("duplicate_of")
    confidence = result["confidence"].get("duplicate_of", 0.0)
    if not isinstance(value, str) or (value != "none" and value not in ids):
        return None
    return value, confidence


def _digest(report: dict[str, Any]) -> dict[str, Any]:
    digested: dict[str, Any] = {}
    for field in _DIGEST_FIELDS:
        value = report.get(field)
        if value:
            text = str(value)
            suffix = "...[truncated]" if len(text) > _FIELD_CHARS else ""
            digested[field] = text[:_FIELD_CHARS] + suffix
    return digested
