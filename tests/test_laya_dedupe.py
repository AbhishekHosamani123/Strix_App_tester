"""Tests for the Laya decision-model integration and dedupe path."""

from __future__ import annotations

import json
from typing import Any

import pytest

from strix.config import loader
from strix.config.settings import LayaSettings, Settings
from strix.laya import dedupe as laya_dedupe
from strix.report.dedupe import check_duplicate


class _FakeEngine:
    """Decides every question with one fixed answer and confidence."""

    def __init__(
        self,
        value: str = "none",
        confidence: float = 1.0,
        *,
        fail: bool = False,
        return_none: bool = False,
    ) -> None:
        self.value = value
        self.confidence = confidence
        self.fail = fail
        self.return_none = return_none
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        if self.fail:
            raise RuntimeError("laya exploded")
        if self.return_none:
            return None
        self.calls.append(
            {
                "state": json.loads(state),
                "enum": schema["properties"]["duplicate_of"]["enum"],
            }
        )
        return {
            "values": {"duplicate_of": self.value},
            "confidence": {"duplicate_of": self.confidence},
        }


def _enable_laya(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    for key in (
        "STRIX_LAYA_ENABLED",
        "STRIX_LAYA_DEDUPE",
        "STRIX_LAYA_MIN_CONFIDENCE",
        "STRIX_LAYA_MODEL",
        "STRIX_LAYA_SUBFOLDER",
        "STRIX_LAYA_DEVICE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"STRIX_LAYA_ENABLED": "true", **kwargs}.items():
        monkeypatch.setenv(key, value)
    loader._cached = None


def _patch_engine(monkeypatch: pytest.MonkeyPatch, engine: _FakeEngine | None) -> None:
    monkeypatch.setattr(laya_dedupe, "get_engine", lambda: engine)


def _report(report_id: str, title: str = "SQL injection") -> dict[str, Any]:
    return {"id": report_id, "title": title, "endpoint": "/api/login", "method": "POST"}


async def test_duplicate_when_confident(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    engine = _FakeEngine(value="vuln-0002", confidence=0.9)
    _patch_engine(monkeypatch, engine)

    result = await laya_dedupe.check_duplicate_with_laya(
        _report("cand"), [_report("vuln-0001"), _report("vuln-0002")]
    )

    assert result == {
        "is_duplicate": True,
        "duplicate_id": "vuln-0002",
        "confidence": 0.9,
        "reason": (
            "Laya decision model: same vulnerability as existing report vuln-0002 (confidence 0.90)"
        ),
    }


async def test_no_duplicate_when_confident_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(value="none", confidence=0.95))

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is not None
    assert result["is_duplicate"] is False
    assert result["duplicate_id"] == ""
    assert result["confidence"] == 0.95


@pytest.mark.parametrize(
    ("value", "confidence"),
    [("vuln-0001", 0.4), ("none", 0.4)],
)
async def test_low_confidence_escalates_to_llm(
    monkeypatch: pytest.MonkeyPatch, value: str, confidence: float
) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(value=value, confidence=confidence))

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None


async def test_engine_unavailable_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, None)

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None


async def test_engine_error_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(fail=True))

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None


async def test_engine_returning_none_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(return_none=True))

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None


async def test_unknown_report_id_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(value="vuln-9999", confidence=0.99))

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None


async def test_laya_disabled_escalates_without_engine() -> None:
    loader._cached = None
    try:
        result = await laya_dedupe.check_duplicate_with_laya(
            _report("cand"), [_report("vuln-0001")]
        )
    finally:
        loader._cached = None
    assert result is None


async def test_dedupe_flag_disabled_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch, STRIX_LAYA_DEDUPE="false")
    engine = _FakeEngine(value="none", confidence=1.0)
    _patch_engine(monkeypatch, engine)

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), [_report("vuln-0001")])

    assert result is None
    assert engine.calls == []


async def test_existing_reports_chunked_per_forward_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_laya(monkeypatch)
    engine = _FakeEngine(value="none", confidence=0.9)
    _patch_engine(monkeypatch, engine)
    reports = [_report(f"vuln-{index:04d}") for index in range(20)]

    result = await laya_dedupe.check_duplicate_with_laya(_report("cand"), reports)

    assert result is not None
    assert result["is_duplicate"] is False
    assert len(engine.calls) == 3  # 8 + 8 + 4
    assert all(len(call["enum"]) <= 32 for call in engine.calls)


async def test_state_digests_truncate_long_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_laya(monkeypatch)
    engine = _FakeEngine(value="none", confidence=0.9)
    _patch_engine(monkeypatch, engine)
    report = _report("vuln-0001")
    report["description"] = "A" * 1000

    await laya_dedupe.check_duplicate_with_laya(_report("cand"), [report])

    digested = engine.calls[0]["state"]["existing_reports"][0]["description"]
    assert len(digested) == laya_dedupe._FIELD_CHARS + len("...[truncated]")


async def test_check_duplicate_routes_to_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(value="none", confidence=0.9))

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("LLM dedupe path must not run when Laya answers")

    monkeypatch.setattr("strix.report.dedupe.resolve_dedupe_model", _fail)

    result = await check_duplicate(_report("cand"), [_report("vuln-0001")])

    assert result["is_duplicate"] is False
    assert "Laya" in result["reason"]


async def test_check_duplicate_falls_back_to_llm_when_laya_escalates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_laya(monkeypatch)
    _patch_engine(monkeypatch, _FakeEngine(value="none", confidence=0.2))

    def _no_llm(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no LLM configured in this test")

    monkeypatch.setattr("strix.report.dedupe.resolve_dedupe_model", _no_llm)

    result = await check_duplicate(_report("cand"), [_report("vuln-0001")])

    # The escalation reached the LLM path, which failed as staged.
    assert result["is_duplicate"] is False
    assert "no LLM configured" in result["reason"]


def test_laya_settings_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LAYA_ENABLED", "true")
    monkeypatch.setenv("STRIX_LAYA_MODEL", "convaiinnovations/laya")
    monkeypatch.setenv("STRIX_LAYA_SUBFOLDER", "multilingual")
    monkeypatch.setenv("STRIX_LAYA_DEVICE", "cpu")
    monkeypatch.setenv("STRIX_LAYA_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("STRIX_LAYA_DEDUPE", "false")

    laya = LayaSettings()

    assert laya.enabled is True
    assert laya.model == "convaiinnovations/laya"
    assert laya.subfolder == "multilingual"
    assert laya.device == "cpu"
    assert laya.min_confidence == 0.5
    assert laya.dedupe is False


def test_laya_settings_defaults() -> None:
    laya = LayaSettings()

    assert laya.enabled is False
    assert laya.model == "convaiinnovations/laya"
    assert laya.subfolder == "typed-decisions"
    assert laya.device is None
    assert laya.min_confidence == 0.7
    assert laya.dedupe is True


def test_settings_carries_laya_section() -> None:
    assert Settings().laya.enabled is False
