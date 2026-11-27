"""Glue between the length-fit building blocks and the Tailor and create flows.

``content_budget`` (the plan), ``budget_acceptance`` (per-entry acceptance of
``rewrite_rows``) and ``length_fit`` (the safety net) are pure and injectable. This
module supplies what they need from a running request: the real renderer, the time
caps, the LLM calls and the bookkeeping that keeps a registered preview's source in
step with the final result (so confirm validates against it).

Order of one tailored resume:

1. ``plan_length``: render once, drop Skills/Awards/Certifications when over, plan the
   budget. Its text goes into the first generation prompt, no extra LLM call.
2. The caller merges the accepted ``rewrite_rows`` entries into the source
   (``accept_rewrite_rows``), then runs the usual writing loop and preservation.
3. ``run_tailor_length_fit`` / ``run_create_length_fit``: measure the finished result
   and fix a miss. Non-AI steps first; at most one extra LLM call per round, three
   rounds, bounded by the remaining operation time.
"""

import copy
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from app.ai_budget import AIOperationDeadlineExceeded, remaining_timeout
from app.ai_limits import PromptSizeError
from app.prompts import LENGTH_STRATEGY_ID
from app.schemas.models import ImproveDiffResult, LengthFitSummary, PageFitSettings
from app.services.budget_acceptance import accept_rewrite_rows, rewrites_from_changes
from app.services.bullet_selector import (
    SELECTABLE_SECTIONS,
    BulletKey,
    PageMeasureError,
)
from app.services.content_budget import (
    ContentBudget,
    plan_budget,
    render_budget_block,
)
from app.services.length_fit import (
    AdditionalDrop,
    LengthFitResult,
    RewriteOutcome,
    RewriteRequest,
    RowRestore,
    additional_drop_order,
    remove_additional_to_fit,
    run_length_fit,
)
from app.services.page_fit import LayoutMeasurement, classify_fill
from app.services.resume_preservation import finalize_ai_resume
from app.services.tailor_selection import measure_layout_within_budget

logger = logging.getLogger(__name__)

# Planning renders may spend at most min(ceiling, share of the remaining budget), so
# the LLM stages after them keep their time (same shape as bullet selection's cap).
PLAN_TOTAL_BUDGET_SECONDS = 60.0
PLAN_BUDGET_SHARE = 0.25

GenerateDiffsFn = Callable[..., Awaitable[ImproveDiffResult]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def resume_description_text(data: dict[str, Any]) -> str:
    """Every work and project row, for "is this skill used in the bullets" checks."""
    rows: list[str] = []
    for section in SELECTABLE_SECTIONS:
        for entry in data.get(section) or []:
            if isinstance(entry, dict):
                rows.extend(
                    row
                    for row in entry.get("description") or []
                    if isinstance(row, str)
                )
    return "\n".join(rows)


def effective_description_format(
    description_format: str, page_fit: PageFitSettings | None
) -> str:
    """ "text" when the request or the print settings ask for it, else "bullet"."""
    if description_format == "text" or (
        page_fit is not None and page_fit.descriptionFormat == "text"
    ):
        return "text"
    return "bullet"


def remap_scores(
    data: dict[str, Any],
    full: dict[str, Any],
    scores: dict[BulletKey, float],
) -> dict[BulletKey, float]:
    """Bullet scores keyed to ``full`` (the master) re-keyed to ``data``, whose rows are
    a subsequence of the master's rows per entry. Rows that do not match score 0."""
    remapped: dict[BulletKey, float] = {}
    for section in SELECTABLE_SECTIONS:
        entries = data.get(section)
        full_entries = full.get(section)
        if not isinstance(entries, list) or not isinstance(full_entries, list):
            continue
        for index, entry in enumerate(entries):
            if index >= len(full_entries) or not isinstance(entry, dict):
                continue
            rows = entry.get("description")
            full_rows = (
                full_entries[index].get("description")
                if isinstance(full_entries[index], dict)
                else None
            )
            if not isinstance(rows, list) or not isinstance(full_rows, list):
                continue
            cursor = 0
            for row_index, row in enumerate(rows):
                for found in range(cursor, len(full_rows)):
                    if full_rows[found] == row:
                        remapped[(section, index, row_index)] = scores.get(
                            (section, index, found), 0.0
                        )
                        cursor = found + 1
                        break
    return remapped


def _plan_deadline() -> float:
    try:
        share = remaining_timeout() * PLAN_BUDGET_SHARE
    except AIOperationDeadlineExceeded:
        share = 0.0
    return monotonic() + min(PLAN_TOTAL_BUDGET_SECONDS, share)


def _measurer(
    page_fit: PageFitSettings, deadline: float | None = None
) -> Callable[[dict[str, Any]], Awaitable[LayoutMeasurement]]:
    """A render-and-measure function bound to one set of print settings."""

    async def measure(candidate: dict[str, Any]) -> LayoutMeasurement:
        if deadline is not None and monotonic() >= deadline:
            raise PageMeasureError("Length-fit render time cap spent")
        return await measure_layout_within_budget(candidate, page_fit)

    return measure


# ---------------------------------------------------------------------------
# Step 1: probe, drop, budget
# ---------------------------------------------------------------------------


@dataclass
class LengthPlan:
    """The pre-generation result: what the first prompt shows and what changed."""

    target_pages: int | None
    description_format: str
    # The resume the budget was planned on: the source minus the Skills, Awards and
    # Certifications dropped to reach the page target.
    data: dict[str, Any]
    budget: ContentBudget | None = None
    layout: LayoutMeasurement | None = None
    dropped: list[AdditionalDrop] = field(default_factory=list)
    pages_before: int | None = None
    fill_before: float | None = None
    renders: int = 0

    @property
    def planned_data(self) -> dict[str, Any]:
        """``data`` without the bullets the budget drops first. The budget's paths and
        row counts refer to this version, and it is what the LLM is shown."""
        return self.budget.apply_drops(self.data) if self.budget else self.data

    @property
    def block(self) -> str:
        """The ``{length_budget}`` text ("" when there is nothing to say)."""
        budget = self.budget
        if budget is None or not (budget.budgeted_paths() or budget.drops):
            return ""  # nothing to change: the prompt stays as it always was
        return render_budget_block(budget)


async def plan_length(
    data: dict[str, Any],
    *,
    page_fit: PageFitSettings | None,
    target_pages: int | None,
    description_format: str,
    job_keywords: dict[str, Any],
    job_description: str,
    scores: dict[BulletKey, float] | None = None,
    drop_skills: bool = True,
) -> LengthPlan:
    """Probe ``data``, drop what the page target needs, plan the content budget.

    With no page target (unlimited) nothing is rendered: the budget keeps every entry's
    length and only restructures into result rows when the format is "text". A render
    that fails or runs out of time leaves the plan without a budget, so the flow
    carries on exactly as it did before length fitting existed. ``drop_skills`` False
    keeps technical skills out of the drop order (create mode writes its own skills).
    """
    plan = LengthPlan(target_pages, description_format, data)
    if target_pages is None or page_fit is None:
        if description_format == "text":
            plan.budget = plan_budget(
                data,
                LayoutMeasurement(pages=0, last_page_fill=0.0),
                None,
                scores=scores,
                job_keywords=job_keywords,
                description_format=description_format,
            )
        return plan

    measure = _measurer(page_fit, _plan_deadline())
    try:
        layout = await measure(data)
    except PageMeasureError as e:
        logger.warning(
            "Content budget skipped; the resume could not be measured: %s", e
        )
        return plan
    plan.renders = 1
    plan.pages_before = layout.pages
    plan.fill_before = round(layout.last_page_fill, 3)

    if classify_fill(layout.pages, layout.last_page_fill, target_pages) == "over":
        order = additional_drop_order(
            data.get("additional") or {},
            job_keywords=job_keywords,
            job_description=job_description,
            resume_text=resume_description_text(data),
        )
        if not drop_skills:
            order = [item for item in order if item.field != "technicalSkills"]
        removal = await remove_additional_to_fit(
            data, order, measure, target_pages, baseline=layout
        )
        plan.renders += removal.renders
        if removal.dropped and removal.layout is not None:
            plan.data = removal.data
            plan.dropped = list(removal.dropped)
            layout = removal.layout
    plan.layout = layout
    plan.budget = plan_budget(
        plan.data,
        layout,
        target_pages,
        scores=scores,
        job_keywords=job_keywords,
        description_format=description_format,
    )
    logger.info(
        "Content budget: status=%s pages=%d fill=%.2f delta_lines=%d dropped=%d "
        "bullet_drops=%d",
        plan.budget.status,
        layout.pages,
        layout.last_page_fill,
        plan.budget.delta_lines,
        len(plan.dropped),
        len(plan.budget.drops),
    )
    return plan


# ---------------------------------------------------------------------------
# Row restore (Tailor): master bullets the page-fit selection trimmed
# ---------------------------------------------------------------------------


def build_row_restore(
    full: dict[str, Any],
    selected: dict[str, Any],
    scores: dict[BulletKey, float],
) -> RowRestore | None:
    """Put trimmed master bullets back, most relevant first, at the end of their entry.

    ``full`` is the master, ``selected`` the data bullet selection kept (its rows are
    copies of master rows). None when nothing was trimmed.
    """
    trimmed: list[tuple[float, int, int, int, str, str]] = []
    for rank, section in enumerate(SELECTABLE_SECTIONS):
        full_entries = full.get(section)
        kept_entries = selected.get(section)
        if not isinstance(full_entries, list) or not isinstance(kept_entries, list):
            continue
        for index, full_entry in enumerate(full_entries):
            if index >= len(kept_entries) or not isinstance(full_entry, dict):
                continue
            kept_entry = kept_entries[index]
            full_rows = full_entry.get("description")
            if not isinstance(full_rows, list) or not isinstance(kept_entry, dict):
                continue
            if kept_entry.get("condensed"):
                continue
            kept_counts = Counter(
                row
                for row in kept_entry.get("description") or []
                if isinstance(row, str)
            )
            styles = full_entry.get("descriptionStyles")
            for row_index, row in enumerate(full_rows):
                if not isinstance(row, str):
                    continue
                if kept_counts[row] > 0:
                    kept_counts[row] -= 1
                    continue
                style = (
                    styles[row_index]
                    if isinstance(styles, list) and row_index < len(styles)
                    else "bullet"
                )
                trimmed.append(
                    (
                        -scores.get((section, index, row_index), 0.0),
                        rank,
                        index,
                        row_index,
                        row,
                        style,
                    )
                )
    if not trimmed:
        return None
    trimmed.sort(key=lambda item: item[:4])

    def build(data: dict[str, Any], count: int) -> dict[str, Any]:
        result = copy.deepcopy(data)
        for _score, rank, index, _row_index, row, style in trimmed[:count]:
            entries = result.get(SELECTABLE_SECTIONS[rank])
            if not isinstance(entries, list) or index >= len(entries):
                continue
            entry = entries[index]
            rows = entry.get("description") if isinstance(entry, dict) else None
            if not isinstance(rows, list) or entry.get("condensed"):
                continue
            styles = entry.get("descriptionStyles")
            styles = list(styles) if isinstance(styles, list) else []
            styles = (styles + ["bullet"] * len(rows))[: len(rows)]
            rows.append(row)
            styles.append(style)
            entry["descriptionStyles"] = styles
        return result

    return RowRestore(total=len(trimmed), build=build)


# ---------------------------------------------------------------------------
# Registered source (so confirm validates against what the preview really is)
# ---------------------------------------------------------------------------


def derive_source(
    source: dict[str, Any], final: dict[str, Any], touched: set[str]
) -> dict[str, Any]:
    """The source the preview is registered with once the safety net has run.

    ``source`` is the data the first generation edited (after accepted ``rewrite_rows``
    and drops). The safety net may remove Skills, Awards and Certifications, rewrite
    whole entries or put master bullets back; those parts of the source follow the
    final result (an entry whose row count changed counts as touched), so the
    preservation check at confirm time stays a no-op for them and still guards
    everything else.
    """
    result = copy.deepcopy(source)
    for section in SELECTABLE_SECTIONS:
        source_entries = result.get(section)
        final_entries = final.get(section)
        if not isinstance(source_entries, list) or not isinstance(final_entries, list):
            continue
        for index, (src, fin) in enumerate(zip(source_entries, final_entries)):
            if not isinstance(src, dict) or not isinstance(fin, dict):
                continue
            src_rows, fin_rows = src.get("description"), fin.get("description")
            if not isinstance(src_rows, list) or not isinstance(fin_rows, list):
                continue
            path = f"{section}[{index}].description"
            if path in touched or len(src_rows) != len(fin_rows):
                src["description"] = copy.deepcopy(fin_rows)
                if "descriptionStyles" in fin:
                    src["descriptionStyles"] = copy.deepcopy(fin["descriptionStyles"])
                else:
                    src.pop("descriptionStyles", None)
    result["additional"] = copy.deepcopy(final.get("additional") or {})
    return result


def _finish_summary(
    summary: LengthFitSummary, plan: LengthPlan, extra_dropped: list[AdditionalDrop]
) -> LengthFitSummary:
    """Fold the planning phase (the render before generation, the Skills dropped then)
    into the safety net's summary."""
    if summary.status != "skipped":
        summary.renders += plan.renders
        if plan.pages_before is not None:
            summary.pages_before = plan.pages_before
            summary.last_page_fill_before = plan.fill_before
        summary.dropped = [d.to_dict() for d in extra_dropped] + summary.dropped
    return summary


# ---------------------------------------------------------------------------
# Step 3 (Tailor): the safety net
# ---------------------------------------------------------------------------


@dataclass
class TailorFitOutcome:
    data: dict[str, Any]
    source: dict[str, Any]
    summary: LengthFitSummary
    layout: LayoutMeasurement | None


async def run_tailor_length_fit(
    final: dict[str, Any],
    *,
    source: dict[str, Any],
    plan: LengthPlan,
    page_fit: PageFitSettings | None,
    job_description: str,
    job_keywords: dict[str, Any],
    language: str,
    master: dict[str, Any] | None,
    generate_diffs: GenerateDiffsFn,
    row_restore: RowRestore | None = None,
) -> TailorFitOutcome:
    """Measure the finished resume and fix a page-target miss (see ``run_length_fit``).

    The LLM round reuses ``generate_diffs`` (``generate_resume_diffs``) with the
    ``length`` strategy: it asks only for ``rewrite_rows`` on the entries a fresh budget
    lists, and ``accept_rewrite_rows`` checks each entry. Returns the final data (made
    canonical against the source it will be registered with), that source and the
    summary. Without a page target, or when the resume cannot be rendered, the data
    comes back unchanged.
    """
    target = plan.target_pages
    if target is None or page_fit is None:
        summary = LengthFitSummary(
            target_pages=None, status="skipped", stopped="unlimited"
        )
        return TailorFitOutcome(final, source, summary, None)

    touched: set[str] = set()
    measure = _measurer(page_fit)

    async def rewrite(request: RewriteRequest) -> RewriteOutcome | None:
        budget = plan_budget(
            request.data,
            request.layout,
            target,
            job_keywords=job_keywords,
            description_format=plan.description_format,
        )
        if budget.status == "fit" or not (budget.budgeted_paths() or budget.drops):
            return None
        work = budget.apply_drops(request.data)
        try:
            diff = await generate_diffs(
                original_resume=json.dumps(work, ensure_ascii=False),
                job_description=job_description,
                job_keywords=job_keywords,
                language=language,
                prompt_id=LENGTH_STRATEGY_ID,
                original_resume_data=work,
                skill_targets=[],
                length_budget=render_budget_block(budget),
            )
        except (AIOperationDeadlineExceeded, PromptSizeError):
            raise
        except Exception as e:
            logger.warning("Length-fit rewrite failed, keeping the best version: %s", e)
            return None
        accepted = accept_rewrite_rows(
            work, rewrites_from_changes(diff.changes), budget, master=master
        )
        if accepted.data == request.data:
            return None
        touched.update(accepted.accepted)
        touched.update(entry.path for entry in budget.entries if entry.dropped_rows)
        return RewriteOutcome(accepted.data, accepted.review_paths)

    result: LengthFitResult = await run_length_fit(
        final,
        measure=measure,
        target_pages=target,
        rewrite=rewrite,
        row_restore=row_restore,
        job_keywords=job_keywords,
        job_description=job_description,
    )
    summary = _finish_summary(result.summary, plan, plan.dropped)
    summary.review_paths = list(dict.fromkeys(summary.review_paths))
    if result.data is final:
        return TailorFitOutcome(final, source, summary, result.layout)

    new_source = derive_source(source, result.data, touched)
    canonical = finalize_ai_resume(
        new_source,
        result.data,
        allow_appended_rows=True,
    )
    return TailorFitOutcome(canonical, new_source, summary, result.layout)


# ---------------------------------------------------------------------------
# Create mode
# ---------------------------------------------------------------------------


CreateRepairFn = Callable[[RewriteRequest], Awaitable[dict[str, Any] | None]]


async def run_create_length_fit(
    created: dict[str, Any],
    *,
    plan: LengthPlan,
    page_fit: PageFitSettings | None,
    job_description: str,
    job_keywords: dict[str, Any],
    repair: CreateRepairFn,
) -> tuple[dict[str, Any], LengthFitSummary, LayoutMeasurement | None]:
    """The create-mode safety net: ``repair`` makes the one LLM call of a round (the
    create repair prompt with the page gap as problems) and returns the new resume, or
    None. Non-AI steps (drop or restore Skills, Awards, Certifications) run first."""
    target = plan.target_pages
    if target is None or page_fit is None:
        return (
            created,
            LengthFitSummary(target_pages=None, status="skipped", stopped="unlimited"),
            None,
        )

    async def rewrite(request: RewriteRequest) -> RewriteOutcome | None:
        try:
            data = await repair(request)
        except (AIOperationDeadlineExceeded, PromptSizeError):
            raise
        except Exception as e:
            logger.warning("Create length-fit repair failed, keeping the best: %s", e)
            return None
        if data is None or data == request.data:
            return None
        return RewriteOutcome(data, [])

    result = await run_length_fit(
        created,
        measure=_measurer(page_fit),
        target_pages=target,
        rewrite=rewrite,
        job_keywords=job_keywords,
        job_description=job_description,
    )
    summary = _finish_summary(result.summary, plan, plan.dropped)
    return result.data, summary, result.layout
