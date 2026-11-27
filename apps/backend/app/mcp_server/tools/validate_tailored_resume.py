from typing import Any

from app.mcp_server.tools._shared import (
    ensure_within_size_limit,
    require_resume,
    require_structured,
    tailoring_report,
)
from app.schemas import ResumeData


async def validate_tailored_resume(
    source_resume_id: str, resume_data: ResumeData, length_fit: bool = False
) -> dict[str, Any]:
    """Dry-run the preservation checks save_tailored_resume enforces. Writes nothing.

    Set length_fit true when you dropped skills, awards, certifications or languages, or kept
    fewer rows, to fit a page target: those drops are then allowed. Added awards or
    certifications, changed personalInfo, company, title or dates, and numbers that are not
    in the master entry are still violations.
    """
    source = await require_resume(source_resume_id)
    candidate = resume_data.model_dump()
    ensure_within_size_limit(candidate)
    return tailoring_report(
        require_structured(source), candidate, length_fit=length_fit
    )
