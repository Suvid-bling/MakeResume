"""Helpers and constants shared by the MCP tools."""

import functools
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, get_args
from uuid import uuid4

import anyio.to_thread
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from app.ai_limits import MAX_SOURCE_CHARACTERS
from app.config import settings
from app.database import DatabaseBusyError, ResumeNotFoundError, db
from app.pdf import render_resume_layout
from app.routers.resumes import (
    _cached_job_keywords,
    _get_original_resume_data,
    _grounding_master_data,
    _hash_job_content,
    _score_tailored_data,
    _validate_confirm_payload,
)
from app.schemas import ResumeData, normalize_resume_data
from app.schemas.models import PageFitSettings
from app.services.ats import compute_ats_score
from app.services.improver import generate_improvements
from app.services.page_fit import (
    page_fit_from_template_settings,
    print_margins,
    resume_print_url,
)
from app.services.refiner import analyze_keyword_gaps, calculate_keyword_match
from app.services.resume_preservation import (
    finalize_ai_resume,
    grounding_review_warnings,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

TEMPLATES: list[str] = list(
    get_args(PageFitSettings.model_fields["template"].annotation)
)
MAX_TITLE_LENGTH = 80
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_DIFF_PATHS = 25

AUTHORING_RULES = [
    "Keep personalInfo, employers, institutions, dates and entry counts identical to the source resume.",
    "Never invent metrics, skills, tools or certifications that the source resume does not support. The one exception is filling an under-filled last page (see the length rules below).",
    "Rewrite, reorder and trim bullet text freely; keep descriptionStyles the same length as description.",
    'Row styles: bullet is an action row, result is the outcome of the action row above it (write it with a "-> " prefix or descriptionStyles "result"; one short line, no repeated action), plain has no marker. The resume\'s descriptionFormat setting decides whether rows render as bullets or as "- " / "-> " text; the content is the same.',
    "Inline HTML in descriptions is limited to <strong>, <em>, <u> and <a>.",
    "Keep existing ids and sectionMeta from the source; omit sectionMeta on new resumes to get defaults.",
    "Keep each work entry's condensed flag as in the source; it only controls how the entry is laid out.",
    "Length rules (page target = max_pages): call plan_content_budget first and write the first draft to its per-entry row and line budget. After tailor_and_verify, fix the page fit from last_page_fill and issues, at most 3 rounds. A last page 90% to 100% full counts as full.",
    "Over the page target, cut in this order: technicalSkills the job does not ask for (least relevant first, a skill the job description names is never dropped), then awards, then certificationsTraining, then condense work entries (shorter rows, then fewer rows, never below one row per entry; if more than 40% of an entry's rows would go, drop its least relevant rows first). Pass length_fit=true to tailor_and_verify or validate_tailored_resume when you drop any of these.",
    "Under the page target (under_page_fill), fill in this order: put back the skills, awards and certifications you dropped, then rows you trimmed from the master, then add rows to work entries under their existing company. Added rows may describe responsibilities and methods the entry's role involved, combined with the job description, and may name tools from the job description, but use only numbers already in that master entry, never add awards or certifications, and do not add those tools to Skills.",
    "Language is never shown on the resume. Leave additional.languages as it is in the source; do not write or rely on it.",
]


class JobKeywords(BaseModel):
    """Requirements the agent extracted from a job description."""

    model_config = ConfigDict(extra="allow")

    company: str = ""
    role: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    key_responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    experience_years: int | None = None
    seniority_level: str | None = None


def guard(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Map retryable database contention to a tool error."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except DatabaseBusyError as e:
            raise ToolError("Database is busy. Retry the same call.") from e
        except ResumeNotFoundError as e:
            raise ToolError("Resume not found.") from e

    return wrapper


async def require_resume(resume_id: str) -> dict[str, Any]:
    resume = await db.get_resume(resume_id)
    if not resume:
        raise ToolError(f"Resume not found: {resume_id}")
    return resume


async def require_job(job_id: str) -> dict[str, Any]:
    job = await db.get_job(job_id)
    if not job:
        raise ToolError(f"Job not found: {job_id}")
    return job


def require_structured(resume: dict[str, Any]) -> dict[str, Any]:
    data = _get_original_resume_data(resume)
    if not data:
        raise ToolError(
            "Resume has no structured data yet. Read its source_markdown with "
            "get_resume, then save structured data with update_resume."
        )
    return data


def require_valid_structured(resume: dict[str, Any]) -> dict[str, Any]:
    """Structured data that passes the schema, else a stable tool error (details are logged)."""
    data = require_structured(resume)
    try:
        ResumeData.model_validate(normalize_resume_data(dict(data)))
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(
            "Stored data of %s does not match the schema: %s", resume["resume_id"], e
        )
        raise ToolError(
            "This resume's stored data is invalid. Rewrite it with update_resume, "
            "or open it in the builder and save it again."
        ) from e
    return data


async def read_document_bytes(path: Path) -> bytes:
    """Read a local file, refusing anything over MAX_DOCUMENT_BYTES.

    Reads at most one byte past the cap from a single opened handle, so a file
    that grows or is swapped after any earlier size check is never loaded in full.
    """

    def read() -> bytes:
        with path.open("rb") as handle:
            return handle.read(MAX_DOCUMENT_BYTES + 1)

    try:
        content = await anyio.to_thread.run_sync(read)
    except OSError as e:
        logger.error("Could not read %s: %s", path, e)
        raise ToolError("Could not read this document.") from e
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ToolError("File too large. Maximum size is 4MB.")
    return content


def ensure_within_size_limit(data: dict[str, Any]) -> None:
    if len(json.dumps(data, ensure_ascii=False)) > MAX_SOURCE_CHARACTERS:
        raise ToolError(f"Resume exceeds {MAX_SOURCE_CHARACTERS} characters.")


def canonical(resume_data: ResumeData) -> dict[str, Any]:
    data = ResumeData.model_validate(
        normalize_resume_data(resume_data.model_dump(mode="json"))
    ).model_dump(mode="json")
    ensure_within_size_limit(data)
    return data


def clean_title(title: str | None) -> str | None:
    cleaned = (title or "").strip()[:MAX_TITLE_LENGTH]
    return cleaned or None


def summary(resume: dict[str, Any]) -> dict[str, Any]:
    return {
        "resume_id": resume["resume_id"],
        "title": resume.get("title"),
        "is_master": resume.get("is_master", False),
        "is_default_master": resume.get("is_default_master", False),
        "parent_id": resume.get("parent_id"),
        "processing_status": resume.get("processing_status"),
        "updated_at": resume.get("updated_at"),
    }


def job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "content": job.get("content"),
        "company": job.get("company"),
        "role": job.get("role"),
        "job_keywords": job.get("job_keywords"),
        "created_at": job.get("created_at"),
    }


def keyword_updates(content: str, keywords: JobKeywords) -> dict[str, Any]:
    # Scoring reads only these three lists; an empty payload would still be cached
    # as truthy and score every resume 0% instead of reporting missing keywords.
    terms = (*keywords.required_skills, *keywords.preferred_skills, *keywords.keywords)
    if not any(term.strip() for term in terms):
        raise ToolError(
            "keywords must include at least one of required_skills, "
            "preferred_skills or keywords."
        )
    updates: dict[str, Any] = {
        "job_keywords": keywords.model_dump(),
        "job_keywords_hash": _hash_job_content(content),
    }
    if keywords.company.strip():
        updates["company"] = keywords.company.strip()
    if keywords.role.strip():
        updates["role"] = keywords.role.strip()
    return updates


def diff_paths(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key in a and key in b:
                out.extend(diff_paths(a[key], b[key], child))
            else:
                out.append(child)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [path or "<root>"]


def tailoring_report(
    original: dict[str, Any], candidate: dict[str, Any], *, length_fit: bool = False
) -> dict[str, Any]:
    """Run the same preservation checks as improve/confirm.

    With ``length_fit`` the candidate was cut or expanded to fit a page target: it may
    drop skills, awards, certifications and languages and keep fewer rows per entry.
    Added awards or certifications, changed personalInfo, company, title or dates, and
    numbers that are not in the master entry are still violations.
    """
    violations: list[str] = []
    report: dict[str, Any] = {}
    repaired = ResumeData.model_validate(
        finalize_ai_resume(
            original, candidate, allow_appended_rows=True, length_fit=length_fit
        )
    ).model_dump()
    if repaired != candidate:
        paths = diff_paths(candidate, repaired)
        violations.append(
            "preservation repairs required at: " + ", ".join(paths[:MAX_DIFF_PATHS])
        )
        report["repaired_resume_data"] = repaired
    try:
        _validate_confirm_payload(
            original, candidate, allow_appended_rows=True, length_fit=length_fit
        )
    except ValueError as e:
        violations.append(str(e))
    changed = diff_paths(original, candidate)
    if violations and not length_fit:
        report["length_fit_hint"] = (
            "If you dropped skills, awards, certifications or languages, or kept fewer "
            "rows, to fit the page target, call again with length_fit=true."
        )
    return {
        "valid": not violations,
        "violations": violations,
        "grounding_warnings": grounding_review_warnings(original, candidate),
        "changed_fields": changed[:MAX_DIFF_PATHS],
        "changed_field_count": len(changed),
        **report,
    }


def render_failed_message() -> str:
    return (
        "PDF rendering failed. Make sure the Resume Matcher frontend is running "
        f"at {settings.frontend_base_url} and can reach the backend."
    )


async def render_pdf(
    resume_id: str, fit: PageFitSettings
) -> tuple[bytes, dict[str, Any] | None]:
    """The PDF GET /resumes/{id}/pdf returns for these settings, plus the layout
    probe's measurement of the page (the print route's ``probe=1`` mode).

    One render gives both. The probe is ``None`` when the print route did not report
    it; the PDF is still returned then.
    """
    url = f"{resume_print_url(resume_id, fit)}&probe=1"
    return await render_resume_layout(url, fit.pageSize, margins=print_margins(fit))


async def saved_template_settings(
    resume: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """The resume's saved formatting and where it came from: its own, else its
    parent's (a tailored resume uses the master's), else none (the defaults)."""
    own = resume.get("template_settings")
    if isinstance(own, dict) and own:
        return own, "own"
    parent_id = resume.get("parent_id")
    if parent_id:
        parent = await db.get_resume(parent_id)
        inherited = parent.get("template_settings") if parent else None
        if isinstance(inherited, dict) and inherited:
            return inherited, "parent"
    return None, "default"


async def resolve_page_fit(
    resume: dict[str, Any], print_settings: PageFitSettings | None
) -> tuple[PageFitSettings, str]:
    """Print settings for a resume: its saved formatting (own, else parent, else the
    defaults), with any field the caller set in ``print_settings`` on top.

    Returns the settings and the source of the saved part (``own``, ``parent`` or
    ``default``).
    """
    saved, source = await saved_template_settings(resume)
    fit = page_fit_from_template_settings(saved)
    if print_settings is not None:
        overrides = {
            name: getattr(print_settings, name)
            for name in print_settings.model_fields_set
        }
        if overrides:
            fit = fit.model_copy(update=overrides)
    return fit, source


def keyword_score(
    data: dict[str, Any], job_keywords: dict[str, Any], master: dict[str, Any]
) -> dict[str, Any]:
    match = calculate_keyword_match(data, job_keywords)
    gaps = analyze_keyword_gaps(job_keywords, data, master)
    ats = compute_ats_score(
        refined_resume=data,
        job_keywords=job_keywords,
        keyword_match_percentage=match,
        missing_keywords=gaps.non_injectable_keywords,
        injectable_keywords=gaps.injectable_keywords,
    )
    return {"keyword_match_percentage": round(match, 1), **ats}


async def store_tailored_resume(
    source: dict[str, Any],
    job: dict[str, Any],
    candidate: dict[str, Any],
    title: str | None,
    track_application: bool,
) -> dict[str, Any]:
    """Save a validated tailored resume, link it to the job and add a tracker card."""
    source_id = source["resume_id"]
    title = clean_title(title) or clean_title(job.get("role"))
    # The builder's last ATS score, as stored when preview confirmation saves a resume.
    ats_record: dict[str, Any] | None = None
    job_keywords = _cached_job_keywords(job)
    if job_keywords:
        try:
            ats_record = _score_tailored_data(
                candidate,
                job_keywords,
                await _grounding_master_data(source) or candidate,
            )
        except Exception:  # noqa: BLE001 - the score is recalculable, never block the save
            logger.warning("ATS score at save failed", exc_info=True)
    tailored = await db.create_tailored_resume(
        request_id=str(uuid4()),
        original_resume_id=source_id,
        job_id=job["job_id"],
        resume_fields={
            "content": json.dumps(candidate, indent=2),
            "content_type": "json",
            "filename": f"tailored_{source.get('filename') or 'resume'}",
            "is_master": False,
            "parent_id": source_id,
            "processed_data": candidate,
            "processing_status": "ready",
            "title": title,
        },
        improvements=generate_improvements(job.get("job_keywords") or {}),
        ats_score=ats_record,
    )
    result = {**summary(tailored), "ats_score": ats_record}
    if track_application:
        try:
            application = await db.create_application(
                job_id=job["job_id"],
                resume_id=tailored["resume_id"],
                master_resume_id=source_id,
                status="applied",
                company=job.get("company"),
                role=title or job.get("role"),
            )
            result["application_id"] = application["application_id"]
        except Exception as e:  # noqa: BLE001 - tracker is non-critical
            logger.warning("Failed to create tracker application: %s", e)
    return result
