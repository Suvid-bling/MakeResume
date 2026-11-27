from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from app.mcp_server.tools._shared import (
    ensure_within_size_limit,
    guard,
    require_job,
    require_resume,
    require_structured,
    store_tailored_resume,
    tailoring_report,
)
from app.schemas import ResumeData


@guard
async def save_tailored_resume(
    source_resume_id: str,
    job_id: str,
    resume_data: ResumeData,
    title: str | None = None,
    track_application: bool = True,
    length_fit: bool = False,
) -> dict[str, Any]:
    """Save a tailored resume you wrote. Rejected if it breaks preservation rules.

    Set length_fit true when you dropped skills, awards, certifications or languages, or kept
    fewer rows, to fit a page target (see tailor_and_verify).
    """
    source = await require_resume(source_resume_id)
    job = await require_job(job_id)
    candidate = resume_data.model_dump()
    ensure_within_size_limit(candidate)
    report = tailoring_report(
        require_structured(source), candidate, length_fit=length_fit
    )
    if not report["valid"]:
        raise ToolError(
            "Tailored resume rejected: "
            + "; ".join(report["violations"])
            + ". Run validate_tailored_resume for details."
        )
    saved = await store_tailored_resume(
        source, job, candidate, title, track_application
    )
    return {**saved, "grounding_warnings": report["grounding_warnings"]}
