import logging
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from app.database import db
from app.mcp_server.tools._shared import (
    render_failed_message,
    render_pdf,
    require_job,
    require_resume,
    require_valid_structured,
    resolve_page_fit,
)
from app.mcp_server.tools.verify_resume import MaxPages
from app.pdf import PDFRenderError
from app.routers.resumes import _scorable_job_keywords
from app.schemas.models import PageFitSettings
from app.services.content_budget import (
    budget_suggestions,
    plan_budget,
    render_budget_block,
)
from app.services.length_fit import additional_drop_order
from app.services.length_flow import resume_description_text
from app.services.page_fit import measure_pdf_layout, merge_layout, page_limit

logger = logging.getLogger(__name__)


async def plan_content_budget(
    resume_id: str,
    job_id: str | None = None,
    print_settings: PageFitSettings | None = None,
    max_pages: MaxPages = None,
) -> dict[str, Any]:
    """Plan how long each work entry should be so the resume fills max_pages. Writes nothing.

    Call it on the resume you will tailor from (usually the master) before you write the
    first draft, then write each work entry to its budget. It renders the resume once with
    its saved formatting (own template_settings, else its parent's, else the defaults;
    fields you set in print_settings override them), measures the real line count and
    line capacity of every description row, and compares the result with max_pages.

    Returns budget (per work entry: mode keep, expand, condense or restructure, target rows,
    target lines, and the longest an action row (at most 2 lines) and a result row (1 line)
    may be, in words, or in characters for Chinese, Japanese and Korean), budget_text (the
    same budget as prose you can follow), suggestions, and, when the resume is over the
    target, skill_drop_order (skills, then awards, then certifications, least relevant
    first; skills the job asks for are never listed). A last page 90% to 100% full counts
    as full. With print_settings.cv_mode true or no max_pages there is no page target and
    every entry keeps its length. job_id (default: the job a tailored resume was saved for)
    makes the budget favor rows that match the job. Needs the web frontend running.
    """
    resume = await require_resume(resume_id)
    data = require_valid_structured(resume)
    fit, settings_source = await resolve_page_fit(resume, print_settings)
    target = page_limit(fit, max_pages)

    if job_id is None:
        improvement = await db.get_improvement_by_tailored_resume(resume_id)
        job_id = improvement["job_id"] if improvement else None
    job: dict[str, Any] | None = await require_job(job_id) if job_id else None
    job_keywords = _scorable_job_keywords(job) if job else None

    try:
        pdf_bytes, probe = await render_pdf(resume_id, fit)
        layout = merge_layout(measure_pdf_layout(pdf_bytes, fit), probe, data)
    except PDFRenderError as e:
        logger.error("Budget render failed for %s: %s", resume_id, e)
        raise ToolError(render_failed_message()) from e
    except Exception as e:  # noqa: BLE001 - a malformed PDF is a render problem
        logger.error("Could not measure the layout of %s: %s", resume_id, e)
        raise ToolError("The rendered PDF could not be measured.") from e

    budget = plan_budget(
        data,
        layout,
        target,
        job_keywords=job_keywords,
        description_format=fit.descriptionFormat,
    )
    result: dict[str, Any] = {
        "resume_id": resume_id,
        "job_id": job_id,
        "target_pages": target,
        "description_format": fit.descriptionFormat,
        "print_settings_source": settings_source,
        "page_count": layout.pages,
        "last_page_fill": round(layout.last_page_fill, 3),
        "status": budget.status,
        "budget": budget.to_dict(),
        "budget_text": render_budget_block(budget, agent_mode=True),
        "suggestions": budget_suggestions(budget),
        "skill_drop_order": [],
    }
    if budget.status == "over":
        additional = data.get("additional")
        result["skill_drop_order"] = [
            drop.to_dict()
            for drop in additional_drop_order(
                additional if isinstance(additional, dict) else {},
                job_keywords=job_keywords,
                job_description=str((job or {}).get("content") or ""),
                resume_text=resume_description_text(data),
            )
        ]
    return result
