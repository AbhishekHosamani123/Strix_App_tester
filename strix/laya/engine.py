"""Load and query the local Laya decision model.

The ``laya`` package (``pip install laya``, optional extra ``strix-agent[laya]``)
is imported lazily and only when Laya is enabled in settings, so environments
without it keep working unchanged: :func:`get_engine` simply returns ``None``
and every decision point escalates to the configured LLM.

Model weights (~800 MB for one checkpoint) download from Hugging Face on first
use and then stay cached locally.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from strix.config import load_settings


if TYPE_CHECKING:
    from strix.config.settings import LayaSettings

logger = logging.getLogger(__name__)

_ENGINE_LOCK = threading.Lock()
_engine: LayaEngine | None = None


def is_laya_enabled() -> bool:
    """Whether Laya decision-making is switched on in settings."""
    return load_settings().laya.enabled


def is_laya_available() -> bool:
    """Whether the optional ``laya`` package is importable."""
    try:
        import laya  # noqa: F401
    except ImportError:
        return False
    return True


class LayaEngine:
    """Process-wide handle on one loaded Laya checkpoint.

    The checkpoint loads lazily on first use (it can take tens of seconds and
    may download weights), is reused for every subsequent decision, and every
    query runs in a worker thread so the async run loop never blocks on torch.
    """

    def __init__(self, settings: LayaSettings) -> None:
        self._settings = settings
        self._agent: Any | None = None
        self._load_lock = threading.Lock()

    def _get_agent(self) -> Any:
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    import laya

                    logger.info(
                        "Loading Laya checkpoint %s (subfolder=%s, device=%s)",
                        self._settings.model,
                        self._settings.subfolder,
                        self._settings.device or "auto",
                    )
                    self._agent = laya.load(
                        self._settings.model,
                        subfolder=self._settings.subfolder,
                        device=self._settings.device,
                    )
        return self._agent

    async def decide(self, state: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one schema-shaped decision about ``state``.

        The schema is the JSON-schema subset Laya understands: an object whose
        properties are enum choices, booleans, or bounded integer scales.
        Returns ``{"values": ..., "confidence": ...}`` or ``None`` when the
        model could not be loaded or the decision failed for any reason —
        callers must treat ``None`` as "escalate to the LLM", never as an
        answer.
        """
        try:
            import laya

            agent = await asyncio.to_thread(self._get_agent)
            result = await asyncio.to_thread(
                laya.decide,
                agent,
                state,
                schema,
                return_details=True,
            )
        except Exception:
            logger.exception("Laya decision failed; escalating to the LLM")
            return None
        return {
            "values": dict(result.values),
            "confidence": {k: float(v) for k, v in result.confidence.items()},
        }


def get_engine() -> LayaEngine | None:
    """Return the process engine when Laya is enabled and importable, else None."""
    global _engine  # noqa: PLW0603
    if not is_laya_enabled() or not is_laya_available():
        return None
    with _ENGINE_LOCK:
        if _engine is None:
            _engine = LayaEngine(load_settings().laya)
        return _engine


def reset_engine() -> None:
    """Drop the cached engine (used by tests and config reloads)."""
    global _engine  # noqa: PLW0603
    with _ENGINE_LOCK:
        _engine = None
