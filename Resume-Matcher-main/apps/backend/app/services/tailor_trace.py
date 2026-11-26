"""Debug trace of the tailoring decisions: the skill plan and each proposed change.

Logs the model's *stated reasoning* (``strategy_notes`` and each ``reason``),
which targets and changes were accepted or rejected, and a short before/after
preview of every change. It never logs the raw model response.

Off by default. Everything goes to the ``app.tailor_trace`` logger at DEBUG;
``TAILOR_TRACE_LOG=true`` turns it on and writes it to stderr (the backend
console) and to ``apps/backend/logs/tailor_trace.log``. The trace contains resume text, so keep it off outside local
debugging.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.schemas import ImproveDiffResult, ResumeChange

TRACE_LOGGER_NAME = "app.tailor_trace"
#: Longest preview of a single before/after value, in characters.
PREVIEW_CHARS = 240

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


#: File name inside ``settings.logs_dir``.
TRACE_LOG_FILENAME = "tailor_trace.log"
_MAX_LOG_BYTES = 5 * 1024 * 1024
_LOG_BACKUPS = 3


def enable_trace_logging(logs_dir: Path | None = None) -> None:
    """Turn the trace on, writing to stderr and ``<logs_dir>/tailor_trace.log``.

    The app installs no root handler, so DEBUG records would otherwise be
    dropped. Propagation is off so a root handler added later does not print
    each trace twice. The file rotates at 5 MB and keeps 3 backups. If the file
    cannot be opened, the trace still goes to stderr. Safe to call more than once.
    """
    trace_logger.setLevel(logging.DEBUG)
    trace_logger.propagate = False
    if any(getattr(h, "_tailor_trace", False) for h in trace_logger.handlers):
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logs_dir is not None:
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    logs_dir / TRACE_LOG_FILENAME,
                    maxBytes=_MAX_LOG_BYTES,
                    backupCount=_LOG_BACKUPS,
                    encoding="utf-8",
                )
            )
        except OSError:
            logging.getLogger(__name__).warning(
                "Cannot write %s in %s; tracing to stderr only",
                TRACE_LOG_FILENAME,
                logs_dir,
                exc_info=True,
            )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._tailor_trace = True  # type: ignore[attr-defined]
        trace_logger.addHandler(handler)


def trace_enabled() -> bool:
    return trace_logger.isEnabledFor(logging.DEBUG)


def _preview(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        text = " | ".join(str(item) for item in value)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "..."
    return text or "-"


def _targets(lines: list[str], title: str, targets: Iterable[Any]) -> None:
    items = [t for t in targets if isinstance(t, dict)]
    lines.append(f"  {title} ({len(items)}):")
    for target in items:
        lines.append(
            f"    - {target.get('skill', '?')} [{target.get('source', '?')}]: "
            f"{_preview(target.get('reason'))}"
        )


def log_original_resume(
    original_resume_data: dict[str, Any] | None,
    *,
    resume_id: str,
    job_id: str,
) -> None:
    """Log the structured resume the tailoring starts from, in full (not truncated)."""
    if not trace_enabled():
        return
    if original_resume_data:
        body = json.dumps(original_resume_data, ensure_ascii=False, indent=2, default=str)
    else:
        body = "(none: no structured data, the full-output fallback will be used)"
    trace_logger.debug(
        "[tailor-trace] original_resume_data resume=%s job=%s\n%s",
        resume_id,
        job_id,
        body,
    )


def log_skill_plan(
    verified_plan: dict[str, Any],
    *,
    resume_id: str,
    job_id: str,
) -> None:
    """Log the verified skill-target plan: strategy, accepted and rejected targets."""
    if not trace_enabled():
        return
    lines = [
        f"[tailor-trace] skill plan resume={resume_id} job={job_id}",
        f"  strategy: {_preview(verified_plan.get('strategy_notes'))}",
    ]
    _targets(lines, "accepted targets", verified_plan.get("accepted") or [])
    _targets(lines, "rejected targets", verified_plan.get("rejected") or [])
    trace_logger.debug("\n".join(lines))


def log_resume_diffs(
    diff_result: ImproveDiffResult,
    applied: Sequence[ResumeChange],
    rejected: Sequence[ResumeChange],
    *,
    resume_id: str,
    job_id: str,
) -> None:
    """Log every proposed change with its reason and whether it was applied."""
    if not trace_enabled():
        return
    applied_ids = {id(change) for change in applied}
    rejected_ids = {id(change) for change in rejected}
    lines = [
        f"[tailor-trace] resume diffs resume={resume_id} job={job_id}: "
        f"{len(diff_result.changes)} proposed, {len(applied)} applied, "
        f"{len(rejected)} rejected",
        f"  strategy: {_preview(diff_result.strategy_notes)}",
    ]
    for index, change in enumerate(diff_result.changes, start=1):
        if id(change) in applied_ids:
            status = "APPLIED"
        elif id(change) in rejected_ids:
            status = "REJECTED"
        else:
            status = "SKIPPED"
        lines.append(f"  #{index} {status} {change.action} {change.path}")
        lines.append(f"      reason: {_preview(change.reason)}")
        if change.action in ("replace", "reorder"):
            lines.append(f"      before: {_preview(change.original)}")
        lines.append(f"      after:  {_preview(change.value)}")
    trace_logger.debug("\n".join(lines))
