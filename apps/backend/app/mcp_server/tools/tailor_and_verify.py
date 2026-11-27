import logging
from typing import Any

from app.mcp_server.tools._shared import (
    ensure_within_size_limit,
    guard,
    require_job,
    require_resume,
    require_structured,
    store_tailored_resume,
    tailoring_report,
)
from app.mcp_server.tools.verify_resume import MaxPages, verify
from app.schemas import ResumeData
from app.schemas.models import PageFitSettings

logger = logging.getLogger(__name__)


@guard
async def tailor_and_verify(
    source_resume_id: str,
    job_id: str,
    resume_data: ResumeData,
    title: str | None = None,
    print_settings: PageFitSettings | None = None,
    max_pages: MaxPages = None,
    track_application: bool = True,
    length_fit: bool = False,
) -> dict[str, Any]:
    """Validate, save and verify a tailored resume you wrote, in one call.

    If validation fails nothing is written and saved is false. Otherwise the resume is saved
    and the result includes the same report as verify_resume. If verification itself fails
    after the save, the result has saved true and verify_error instead of the report: do not
    save again, call verify_resume. Fix issues with update_resume, then call verify_resume.

    Print settings default to the source resume's saved formatting (its own template_settings,
    else its parent's, else the defaults); fields you set in print_settings override them.
    max_pages is the page target. Set length_fit true when you dropped skills, awards,
    certifications or languages, or kept fewer rows, to fit the page target: those drops are
    then allowed. Added awards or certifications, changed personalInfo, company, title or
    dates, and numbers that are not in the master entry are still rejected. Use the
    last_page_fill and issues (over_page_limit, under_page_fill) in the result to fix the page
    fit, at most 3 rounds.
    """
    source = await require_resume(source_resume_id)
    job = await require_job(job_id)
    candidate = resume_data.model_dump()
    ensure_within_size_limit(candidate)
    validation = tailoring_report(
        require_structured(source), candidate, length_fit=length_fit
    )
    if not validation["valid"]:
        return {"saved": False, "validation": validation}
    saved = await store_tailored_resume(
        source, job, candidate, title, track_application
    )
    try:
        tailored = await require_resume(saved["resume_id"])
        report = await verify(tailored, job_id, print_settings, max_pages)
    except Exception as e:  # noqa: BLE001 - the resume is already saved
        logger.error("Verify failed after saving %s: %s", saved["resume_id"], e)
        return {
            "saved": True,
            "resume": saved,
            "validation": validation,
            "verify_error": (
                "Verification failed after the resume was saved. Do not save "
                "again; call verify_resume with the saved resume_id."
            ),
        }
    return {"saved": True, "resume": saved, "validation": validation, **report}
