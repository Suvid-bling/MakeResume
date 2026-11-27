"""Length safety net: make the final resume fill its page target.

After the first rewrite (planned by ``content_budget``) and the writing loop, the
result is measured once more. This module decides what to do about it, cheapest first:

* Over the target: drop Skills the job does not ask for, then Awards, then
  Certifications, a few at a time, confirmed by real renders (``remove_additional_to_fit``).
* Under the target: put those items back (``restore_order``), then put back master
  bullets trimmed earlier, then ask the LLM to expand.
* Still not full: one extra LLM call per round, at most three rounds, bounded by the
  operation's remaining time. The best measured version always wins.

The LLM and the renderer are injected, so this module has no network or browser
dependency of its own.
"""

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.ai_budget import AIOperationDeadlineExceeded, remaining_timeout
from app.schemas.models import (
    SKILL_ROW_BREAK,
    LengthFitSummary,
    _skill_line_kind,
    _split_skill_items,
)
from app.services.bullet_selector import PageMeasureError
from app.services.content_budget import TARGET_FILL
from app.services.page_fit import FillStatus, LayoutMeasurement, classify_fill
from app.services.text_match import keyword_in_text, normalize_skill_key

logger = logging.getLogger(__name__)

AdditionalField = Literal["technicalSkills", "awards", "certificationsTraining"]
MeasureFn = Callable[[dict[str, Any]], Awaitable[LayoutMeasurement]]

MAX_ROUNDS = 3
MAX_SEARCH_RENDERS = 4
# Operation time kept back for the stages after the safety net, and the least time an
# LLM round is worth starting with.
RESERVE_SECONDS = 15.0
MIN_ROUND_SECONDS = 20.0

_JD_SKILL_KEYS = ("required_skills", "preferred_skills", "keywords")
_LABEL_COLONS = (":", "：")
_LIST_FIELDS: tuple[AdditionalField, AdditionalField] = (
    "awards",
    "certificationsTraining",
)


@dataclass(frozen=True)
class AdditionalDrop:
    """One droppable item of the Additional section, addressed in the original list.

    ``skill`` is set when the item is one skill inside a ``Label: a, b`` line; the line
    itself loses only that skill. Otherwise the whole line at ``index`` goes.
    """

    field: AdditionalField
    index: int
    skill: str | None
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "text": self.text}


def _strings(value: Any) -> list[str]:
    return (
        [item for item in value if isinstance(item, str)]
        if isinstance(value, list)
        else []
    )


def _jd_terms(job_keywords: dict[str, Any] | None) -> set[str]:
    terms: set[str] = set()
    for key in _JD_SKILL_KEYS:
        values = (job_keywords or {}).get(key)
        if isinstance(values, list):
            terms.update(
                normalize_skill_key(v)
                for v in values
                if isinstance(v, str) and v.strip()
            )
    return terms


def is_protected_skill(skill: str, terms: set[str], job_description: str) -> bool:
    """A skill the job asks for: listed as required/preferred (or a keyword), or named
    anywhere in the job description. Such skills are never dropped."""
    key = normalize_skill_key(skill)
    if not key:
        return False
    if key in terms:
        return True
    if any(
        keyword_in_text(term, skill) or keyword_in_text(skill, term) for term in terms
    ):
        return True
    return bool(job_description) and keyword_in_text(skill, job_description)


def _overlap(text: str, job_description: str) -> int:
    """How many words of ``text`` the job description also uses."""
    if not job_description:
        return 0
    return sum(
        1
        for word in set(text.casefold().split())
        if len(word) >= 3 and keyword_in_text(word, job_description)
    )


def _skill_units(items: list[str]) -> list[tuple[int, str | None, str]]:
    """``(line index, skill inside a labelled line or None, text)`` for each skill."""
    units: list[tuple[int, str | None, str]] = []
    for index, line in enumerate(items):
        kind = _skill_line_kind(line)
        if kind == "skill":
            units.append((index, None, line.strip()))
        elif kind == "label":
            _label, rest = _split_label(line)
            units.extend((index, skill, skill) for skill in _split_skill_items(rest))
    return units


def _split_label(line: str) -> tuple[str, str]:
    """``"Label: a, b"`` into ``("Label", "a, b")``."""
    stripped = line.strip()
    positions = [stripped.find(colon) for colon in _LABEL_COLONS if colon in stripped]
    cut = min(positions)
    return stripped[:cut].strip(), stripped[cut + 1 :].strip()


def additional_drop_order(
    additional: dict[str, Any],
    *,
    job_keywords: dict[str, Any] | None = None,
    job_description: str = "",
    resume_text: str = "",
) -> list[AdditionalDrop]:
    """Items to drop, in the order to drop them: technical skills first, then awards,
    then certifications. Languages are never listed (they are not shown).

    Skills the job asks for are protected and absent from the list. The rest go least
    useful first: skills the resume's own bullets never mention, then those with the
    least overlap with the job description, then the last listed. A grouped skills line
    (``Label: a, b``) is dropped per skill. Awards and certifications go the same way,
    least job-related first.
    """
    terms = _jd_terms(job_keywords)
    order: list[AdditionalDrop] = []

    skills = _strings(additional.get("technicalSkills"))
    ranked_skills: list[tuple[tuple[int, int, int, int], AdditionalDrop]] = []
    for position, (index, skill, text) in enumerate(_skill_units(skills)):
        if is_protected_skill(text, terms, job_description):
            continue
        used = 1 if resume_text and keyword_in_text(text, resume_text) else 0
        key = (used, _overlap(text, job_description), -index, -position)
        ranked_skills.append(
            (key, AdditionalDrop("technicalSkills", index, skill, text))
        )
    order.extend(drop for _key, drop in sorted(ranked_skills, key=lambda item: item[0]))

    for field_name in _LIST_FIELDS:
        items = _strings(additional.get(field_name))
        ranked = sorted(
            range(len(items)),
            key=lambda i: (_overlap(items[i], job_description), -i),
        )
        order.extend(
            AdditionalDrop(field_name, i, None, items[i].strip())
            for i in ranked
            if items[i].strip()
        )
    return order


def _clean_skill_rows(lines: list[str]) -> list[str]:
    """Remove labels left without skills, then empty rows (stray or doubled breaks)."""
    kept: list[str] = []
    for position, line in enumerate(lines):
        if _skill_line_kind(line) == "bare":
            following = lines[position + 1] if position + 1 < len(lines) else None
            if following is None or _skill_line_kind(following) in (
                "break",
                "label",
                "bare",
            ):
                continue
        kept.append(line)
    cleaned: list[str] = []
    for line in kept:
        if line.strip() == SKILL_ROW_BREAK and (
            not cleaned or cleaned[-1].strip() == SKILL_ROW_BREAK
        ):
            continue
        cleaned.append(line)
    while cleaned and cleaned[-1].strip() == SKILL_ROW_BREAK:
        cleaned.pop()
    return cleaned


def remove_additional(
    additional: dict[str, Any], drops: list[AdditionalDrop] | tuple[AdditionalDrop, ...]
) -> dict[str, Any]:
    """A copy of ``additional`` without ``drops``. Indexes refer to ``additional`` as
    given, so pass the full original list and every drop made so far."""
    result = copy.deepcopy(additional)
    whole: dict[str, set[int]] = {}
    inline: dict[int, set[str]] = {}
    for drop in drops:
        if drop.field == "technicalSkills" and drop.skill is not None:
            inline.setdefault(drop.index, set()).add(normalize_skill_key(drop.skill))
        else:
            whole.setdefault(drop.field, set()).add(drop.index)

    for field_name in _LIST_FIELDS:
        items = result.get(field_name)
        if isinstance(items, list) and field_name in whole:
            result[field_name] = [
                item for i, item in enumerate(items) if i not in whole[field_name]
            ]

    skills = result.get("technicalSkills")
    if isinstance(skills, list) and (whole.get("technicalSkills") or inline):
        rebuilt: list[str] = []
        for index, line in enumerate(skills):
            if index in whole.get("technicalSkills", set()):
                continue
            if index in inline and isinstance(line, str):
                label, rest = _split_label(line)
                remaining = [
                    skill
                    for skill in _split_skill_items(rest)
                    if normalize_skill_key(skill) not in inline[index]
                ]
                if not remaining:
                    continue
                colon = "：" if "：" in line and ":" not in line else ":"
                joiner = "，" if "，" in rest else ", "
                rebuilt.append(f"{label}{colon} {joiner.join(remaining)}")
                continue
            rebuilt.append(line)
        result["technicalSkills"] = _clean_skill_rows(rebuilt)
    return result


def restore_order(dropped: list[AdditionalDrop]) -> list[AdditionalDrop]:
    """Order to put dropped items back: the reverse of the drop order, so the most
    valuable item (dropped last) returns first."""
    return list(reversed(dropped))


def _with_additional(
    data: dict[str, Any], additional: dict[str, Any]
) -> dict[str, Any]:
    result = dict(data)
    result["additional"] = additional
    return result


def _is_over(layout: LayoutMeasurement, target_pages: int) -> bool:
    return classify_fill(layout.pages, layout.last_page_fill, target_pages) == "over"


@dataclass
class RemovalResult:
    """Outcome of ``remove_additional_to_fit``: the data to continue with, every drop
    now in force, the measurement of that data and how many renders it took."""

    data: dict[str, Any]
    dropped: list[AdditionalDrop]
    layout: LayoutMeasurement | None
    renders: int
    over: bool


async def remove_additional_to_fit(
    data: dict[str, Any],
    order: list[AdditionalDrop],
    measure: MeasureFn,
    target_pages: int,
    *,
    additional_full: dict[str, Any] | None = None,
    dropped: list[AdditionalDrop] | None = None,
    baseline: LayoutMeasurement | None = None,
    max_renders: int = MAX_SEARCH_RENDERS,
) -> RemovalResult:
    """Drop the fewest items of ``order`` that bring the resume to the page target.

    ``order`` comes from ``additional_drop_order``. Page count only falls as items go,
    so this is a binary search, with at most ``max_renders`` renders (``baseline``, the
    measurement of ``data``, is not one of them). Everything in ``order`` is dropped
    when even that is not enough: the caller then condenses work entries. A render that
    fails ends the search with the smallest fitting drop measured so far.

    ``additional_full`` is the Additional section before any drop (default: the one in
    ``data``) and ``dropped`` the drops already applied to it, so indexes stay valid.
    """
    full = (
        additional_full
        if additional_full is not None
        else copy.deepcopy(data.get("additional") or {})
    )
    already = list(dropped or [])
    renders = 0

    def build(count: int) -> tuple[dict[str, Any], list[AdditionalDrop]]:
        drops = [*already, *order[:count]]
        return _with_additional(data, remove_additional(full, drops)), drops

    if baseline is not None and not _is_over(baseline, target_pages):
        return RemovalResult(data, already, baseline, 0, False)
    if not order:
        return RemovalResult(data, already, baseline, 0, True)

    async def render(count: int) -> LayoutMeasurement | None:
        nonlocal renders
        renders += 1
        try:
            return await measure(build(count)[0])
        except PageMeasureError as e:
            logger.warning("Dropping skills/awards stopped; render failed: %s", e)
            return None

    top = len(order)
    top_layout = await render(top)
    if top_layout is None:
        return RemovalResult(data, already, baseline, renders, True)
    if _is_over(top_layout, target_pages):
        built, drops = build(top)
        return RemovalResult(built, drops, top_layout, renders, True)
    low, high, high_layout = 0, top, top_layout
    while high - low > 1 and renders < max_renders:
        mid = (low + high) // 2
        mid_layout = await render(mid)
        if mid_layout is None:
            break
        if _is_over(mid_layout, target_pages):
            low = mid
        else:
            high, high_layout = mid, mid_layout
    built, drops = build(high)
    return RemovalResult(built, drops, high_layout, renders, False)


async def restore_additional_to_fit(
    data: dict[str, Any],
    additional_full: dict[str, Any],
    dropped: list[AdditionalDrop],
    measure: MeasureFn,
    target_pages: int,
    *,
    baseline: LayoutMeasurement | None = None,
    max_renders: int = MAX_SEARCH_RENDERS,
) -> RemovalResult:
    """Put back as many dropped items as still keep the resume within the page target
    (``restore_order``: most valuable first), by binary search with at most
    ``max_renders`` renders. The result's ``dropped`` is what stays dropped."""
    order = restore_order(dropped)
    renders = 0

    def build(count: int) -> tuple[dict[str, Any], list[AdditionalDrop]]:
        back = order[:count]
        remaining = [drop for drop in dropped if drop not in back]
        return _with_additional(
            data, remove_additional(additional_full, remaining)
        ), remaining

    if not order:
        return RemovalResult(data, [], baseline, 0, False)

    async def render(count: int) -> LayoutMeasurement | None:
        nonlocal renders
        renders += 1
        try:
            return await measure(build(count)[0])
        except PageMeasureError as e:
            logger.warning("Restoring skills/awards stopped; render failed: %s", e)
            return None

    top = len(order)
    top_layout = await render(top)
    if top_layout is None:
        return RemovalResult(data, list(dropped), baseline, renders, False)
    if not _is_over(top_layout, target_pages):
        built, remaining = build(top)
        return RemovalResult(built, remaining, top_layout, renders, False)
    low, low_layout, high = 0, baseline, top
    while high - low > 1 and renders < max_renders:
        mid = (low + high) // 2
        mid_layout = await render(mid)
        if mid_layout is None:
            break
        if _is_over(mid_layout, target_pages):
            high = mid
        else:
            low, low_layout = mid, mid_layout
    built, remaining = build(low)
    return RemovalResult(built, remaining, low_layout, renders, False)


def fit_distance(layout: LayoutMeasurement, target_pages: int) -> tuple[int, float]:
    """How far a measured version is from the target; smaller is better.

    Fits come first, then versions that are too short, then versions over the target
    (which must always be cut), each ordered by how far off they are in pages.
    """
    used = (layout.pages - 1) + layout.last_page_fill
    goal = (target_pages - 1) + TARGET_FILL
    status = classify_fill(layout.pages, layout.last_page_fill, target_pages)
    if status == "fit":
        return 0, abs(used - goal)
    if status == "under":
        return 1, goal - used
    return 2, used - goal


@dataclass(frozen=True)
class RowRestore:
    """Master bullets trimmed earlier, ranked by relevance.

    ``build(data, count)`` returns ``data`` with the ``count`` most relevant trimmed
    bullets back (monotone in ``count``, 0 <= count <= ``total``). It is tried once per
    run, on the best version at that point.
    """

    total: int
    build: Callable[[dict[str, Any], int], dict[str, Any]]


@dataclass(frozen=True)
class RewriteRequest:
    """What an LLM round gets: the best version so far, its measurement and the gap."""

    round: int
    data: dict[str, Any]
    layout: LayoutMeasurement
    status: FillStatus
    target_pages: int
    # Pages still to add (+) or remove (-) to reach a last page TARGET_FILL full.
    page_gap: float


@dataclass
class RewriteOutcome:
    """Result of an LLM round: the new resume (rows already accepted) and any rows
    that need a human look."""

    data: dict[str, Any]
    review_paths: list[str] = field(default_factory=list)


RewriteFn = Callable[[RewriteRequest], Awaitable[RewriteOutcome | None]]


@dataclass
class _Version:
    data: dict[str, Any]
    layout: LayoutMeasurement
    dropped: list[AdditionalDrop]
    restored_rows: int
    review_paths: list[str]


@dataclass
class LengthFitResult:
    """The best version found, its measurement (None if nothing could be rendered), the
    drops in force on its Additional section and the summary for the preview."""

    data: dict[str, Any]
    layout: LayoutMeasurement | None
    dropped: list[AdditionalDrop]
    summary: LengthFitSummary


def _resume_text(data: dict[str, Any]) -> str:
    rows: list[str] = []
    for section in ("workExperience", "personalProjects"):
        for entry in data.get(section) or []:
            if isinstance(entry, dict):
                rows.extend(_strings(entry.get("description")))
    return "\n".join(rows)


def _summary_status(
    layout: LayoutMeasurement | None, target: int
) -> Literal["fit", "over", "under", "unavailable"]:
    if layout is None:
        return "unavailable"
    return classify_fill(layout.pages, layout.last_page_fill, target)


async def run_length_fit(
    data: dict[str, Any],
    *,
    measure: MeasureFn,
    target_pages: int | None,
    rewrite: RewriteFn | None = None,
    row_restore: RowRestore | None = None,
    additional_full: dict[str, Any] | None = None,
    dropped: list[AdditionalDrop] | None = None,
    job_keywords: dict[str, Any] | None = None,
    job_description: str = "",
    initial_layout: LayoutMeasurement | None = None,
    max_rounds: int = MAX_ROUNDS,
    reserve_seconds: float = RESERVE_SECONDS,
    time_left: Callable[[], float] = remaining_timeout,
) -> LengthFitResult:
    """Measure the finished resume and, when it misses the page target, fix it.

    Each round runs the non-AI steps first (over: drop skills, awards, certifications;
    under: restore them, then ``row_restore`` bullets) and only when the resume still
    misses the target makes ONE ``rewrite`` call (condense when over, expand when under).
    At most ``max_rounds`` rounds run, and none starts without ``MIN_ROUND_SECONDS`` of
    ``time_left()`` beyond ``reserve_seconds``. Every measured version competes
    (``fit_distance``) and the best is returned, so an LLM round that makes things worse
    costs nothing but its time.

    ``additional_full`` is the Additional section before any drop (default: the one in
    ``data``) and ``dropped`` what ``data`` already lacks relative to it.
    """
    full = (
        additional_full
        if additional_full is not None
        else copy.deepcopy(data.get("additional") or {})
    )
    initial_dropped = list(dropped or [])
    renders = 0
    ai_calls = 0

    if target_pages is None:
        return LengthFitResult(
            data,
            None,
            initial_dropped,
            LengthFitSummary(target_pages=None, status="skipped", stopped="unlimited"),
        )

    async def metered(candidate: dict[str, Any]) -> LayoutMeasurement:
        nonlocal renders
        renders += 1
        return await measure(candidate)

    summary = LengthFitSummary(target_pages=target_pages)
    layout0 = initial_layout
    if layout0 is None:
        try:
            layout0 = await metered(data)
        except PageMeasureError as e:
            logger.warning("Length fit skipped; resume could not be measured: %s", e)
            summary.status = "unavailable"
            summary.stopped = "unmeasurable"
            summary.renders = renders
            return LengthFitResult(data, None, initial_dropped, summary)
        except AIOperationDeadlineExceeded:
            summary.status = "unavailable"
            summary.stopped = "timeout"
            summary.renders = renders
            return LengthFitResult(data, None, initial_dropped, summary)

    summary.pages_before = layout0.pages
    summary.last_page_fill_before = round(layout0.last_page_fill, 3)

    best = _Version(data, layout0, initial_dropped, 0, [])
    stopped: Literal[
        "fit", "max_rounds", "timeout", "unmeasurable", "no_progress", "unlimited"
    ] = "max_rounds"
    rounds = 0

    def consider(candidate: _Version) -> None:
        nonlocal best
        if fit_distance(candidate.layout, target_pages) < fit_distance(
            best.layout, target_pages
        ):
            best = candidate

    rows_tried = False
    settled: _Version | None = None

    async def non_ai_pass(status: FillStatus) -> None:
        """The cheap steps: drop items when over, put items and bullets back when under."""
        nonlocal rows_tried
        current = _Version(
            best.data,
            best.layout,
            list(best.dropped),
            best.restored_rows,
            list(best.review_paths),
        )
        if status == "over":
            order = [
                item
                for item in additional_drop_order(
                    full,
                    job_keywords=job_keywords,
                    job_description=job_description,
                    resume_text=_resume_text(current.data),
                )
                if item not in current.dropped
            ]
            removal = await remove_additional_to_fit(
                current.data,
                order,
                metered,
                target_pages,
                additional_full=full,
                dropped=current.dropped,
                baseline=current.layout,
            )
            if removal.layout is not None and removal.data is not current.data:
                consider(
                    _Version(
                        removal.data,
                        removal.layout,
                        removal.dropped,
                        current.restored_rows,
                        current.review_paths,
                    )
                )
        else:
            if current.dropped:
                restoration = await restore_additional_to_fit(
                    current.data,
                    full,
                    current.dropped,
                    metered,
                    target_pages,
                    baseline=current.layout,
                )
                if restoration.layout is not None and len(restoration.dropped) < len(
                    current.dropped
                ):
                    consider(
                        _Version(
                            restoration.data,
                            restoration.layout,
                            restoration.dropped,
                            current.restored_rows,
                            current.review_paths,
                        )
                    )
            after_restore = classify_fill(
                best.layout.pages, best.layout.last_page_fill, target_pages
            )
            if (
                after_restore == "under"
                and row_restore is not None
                and row_restore.total > 0
                and not rows_tried
            ):
                rows_tried = True
                await _restore_rows(best, row_restore, metered, target_pages, consider)

    try:
        for round_no in range(max_rounds + 1):
            status = classify_fill(
                best.layout.pages, best.layout.last_page_fill, target_pages
            )
            if status == "fit":
                stopped = "fit"
                break

            if best is not settled:
                await non_ai_pass(status)
                settled = best

            status = classify_fill(
                best.layout.pages, best.layout.last_page_fill, target_pages
            )
            if status == "fit":
                stopped = "fit"
                break
            if round_no == max_rounds:
                stopped = "max_rounds"
                break
            if rewrite is None:
                stopped = "no_progress"
                break
            try:
                budget = time_left() - reserve_seconds
            except AIOperationDeadlineExceeded:
                stopped = "timeout"
                break
            if budget < MIN_ROUND_SECONDS:
                stopped = "timeout"
                break

            # One LLM call for this round, from the best version so far.
            used = (best.layout.pages - 1) + best.layout.last_page_fill
            request = RewriteRequest(
                round=round_no + 1,
                data=best.data,
                layout=best.layout,
                status=status,
                target_pages=target_pages,
                page_gap=(target_pages - 1) + TARGET_FILL - used,
            )
            rounds += 1
            ai_calls += 1
            try:
                outcome = await asyncio.wait_for(rewrite(request), timeout=budget)
            except TimeoutError:
                logger.warning("Length fit round %d ran out of time", rounds)
                stopped = "timeout"
                break
            if outcome is None:
                stopped = "no_progress"
                break
            candidate_layout = await metered(outcome.data)
            consider(
                _Version(
                    outcome.data,
                    candidate_layout,
                    list(best.dropped),
                    best.restored_rows,
                    [*best.review_paths, *outcome.review_paths],
                )
            )
    except PageMeasureError as e:
        logger.warning("Length fit stopped; a render failed: %s", e)
        stopped = "unmeasurable"
    except AIOperationDeadlineExceeded:
        stopped = "timeout"

    dropped_now = {(d.field, d.index, d.skill) for d in best.dropped}
    summary.pages_after = best.layout.pages
    summary.last_page_fill_after = round(best.layout.last_page_fill, 3)
    summary.status = _summary_status(best.layout, target_pages)
    summary.renders = renders
    summary.rounds = rounds
    summary.ai_calls = ai_calls
    summary.dropped = [d.to_dict() for d in best.dropped]
    summary.restored = [
        d.to_dict()
        for d in initial_dropped
        if (d.field, d.index, d.skill) not in dropped_now
    ]
    if best.restored_rows:
        summary.restored.append(
            {"field": "workExperience", "text": f"{best.restored_rows} bullet(s)"}
        )
    summary.review_paths = list(dict.fromkeys(best.review_paths))
    summary.stopped = stopped
    return LengthFitResult(best.data, best.layout, best.dropped, summary)


async def _restore_rows(
    best: _Version,
    row_restore: RowRestore,
    measure: MeasureFn,
    target_pages: int,
    consider: Callable[[_Version], None],
    max_renders: int = MAX_SEARCH_RENDERS,
) -> None:
    """Binary-search the most trimmed master bullets that can come back without going
    over the page target (at most ``max_renders`` renders)."""
    renders = 0
    low, low_version, high = 0, None, row_restore.total

    async def render(count: int) -> _Version | None:
        nonlocal renders
        renders += 1
        data = row_restore.build(best.data, count)
        try:
            layout = await measure(data)
        except PageMeasureError as e:
            logger.warning("Restoring bullets stopped; render failed: %s", e)
            return None
        return _Version(
            data, layout, list(best.dropped), count, list(best.review_paths)
        )

    top = await render(row_restore.total)
    if top is None:
        return
    if not _is_over(top.layout, target_pages):
        consider(top)
        return
    while high - low > 1 and renders < max_renders:
        mid = (low + high) // 2
        version = await render(mid)
        if version is None:
            break
        if _is_over(version.layout, target_pages):
            high = mid
        else:
            low, low_version = mid, version
    if low_version is not None:
        consider(low_version)
