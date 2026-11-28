"""Content budget: how many lines each work entry gets, planned before generation.

Pure code, no LLM. ``plan_budget`` takes one measured render of the resume
(``LayoutMeasurement``) and turns the gap to the page target into a target row and
line count per work entry. ``render_budget_block`` turns that plan into the text that
goes into the first generation prompt, so expanding, condensing and splitting rows
into action and result rows happen in that same call.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.bullet_selector import (
    BulletKey,
    bullet_path,
    drop_bullets,
    iter_bullet_rows,
    trim_order,
)
from app.services.page_fit import (
    FILL_MIN,
    LayoutMeasurement,
    classify_fill,
    row_text_key,
)
from app.services.text_match import _CJK_RE, keyword_in_text

logger = logging.getLogger(__name__)

BudgetMode = Literal["keep", "expand", "condense", "restructure"]
BudgetStatus = Literal["fit", "over", "under", "unlimited"]

# Where the last page should end after the rewrite (inside the 90% to 100% tolerance).
TARGET_FILL = 0.95
# A line is planned a little shorter than it measured, so a row at its cap still fits.
LINE_SAFETY = 0.9
ACTION_MAX_LINES = 2
RESULT_MAX_LINES = 1
# Expanding adds at most max(EXPAND_MIN_ROWS, EXPAND_SHARE of the existing rows) rows.
EXPAND_MIN_ROWS = 3
EXPAND_SHARE = 0.5
# Condensing removes at most this share of an entry's rows; deeper cuts drop bullets.
CONDENSE_ROW_SHARE = 0.4
MIN_ROWS = 1
# Share of an entry's rows that become result rows in the text format.
RESULT_ROW_SHARE = 0.4
# Rows may land this far from the target row count (matches the +-1 style check).
ROW_TOLERANCE = 1

# Used when the page has no measurement to learn from.
DEFAULT_LINE_UNITS = {False: 16, True: 36}
DEFAULT_LINES_PER_PAGE = 50.0
MIN_LINE_UNITS = 4

_JOB_TERM_KEYS = ("required_skills", "preferred_skills", "keywords")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_PRESENT_RE = re.compile(
    r"present|current|now|至今|现在|現在|目前|현재|actuel|actual|presente|heute",
    re.IGNORECASE,
)
_WORK_PATH_RE = re.compile(r"^workExperience\[(\d+)\]\.description$")


def is_cjk_text(text: str) -> bool:
    """True when more than 30% of the non-space characters are CJK (as the probe)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(1 for c in chars if _CJK_RE.match(c)) / len(chars) > 0.3


def count_units(text: str, cjk: bool) -> int:
    """Length of a row in the unit the budget uses: characters for CJK text, words
    for everything else. Matches the layout probe's ``countUnits``."""
    plain = row_text_key(text)
    if cjk:
        return len("".join(plain.split()))
    return len(plain.split())


@dataclass(frozen=True)
class EntryBudget:
    """The plan for one work entry. Row and line counts are after ``drops``."""

    path: str  # workExperience[i].description
    entry_index: int
    label: str
    mode: BudgetMode
    current_rows: int
    current_lines: int
    target_rows: int
    target_lines: int
    min_rows: int
    max_rows: int
    result_rows: int  # result rows wanted in the entry (0 = none needed)
    cjk: bool
    line_units: int  # words or characters that fill one line
    action_max_units: int  # an action row: at most ACTION_MAX_LINES lines
    result_max_units: int  # a result row: at most RESULT_MAX_LINES line
    weight: float
    dropped_rows: tuple[int, ...] = ()  # source row indexes removed before rewriting

    @property
    def unit(self) -> str:
        return "characters" if self.cjk else "words"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "mode": self.mode,
            "current_rows": self.current_rows,
            "current_lines": self.current_lines,
            "target_rows": self.target_rows,
            "target_lines": self.target_lines,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "result_rows": self.result_rows,
            "unit": self.unit,
            "line_units": self.line_units,
            "action_max_units": self.action_max_units,
            "result_max_units": self.result_max_units,
            "dropped_rows": list(self.dropped_rows),
        }


@dataclass(frozen=True)
class ContentBudget:
    """What the page needs and how each work entry should change to supply it."""

    target_pages: int | None
    status: BudgetStatus
    pages: int
    last_page_fill: float
    text_format: bool
    delta_lines: int  # lines the rewrite has to add (+) or remove (-)
    planned_lines: int  # what the entries below cover of delta_lines
    lines_per_page: float
    reliable: bool  # False when the layout probe gave no line measurements
    entries: tuple[EntryBudget, ...] = ()
    drops: tuple[BulletKey, ...] = ()  # bullets to drop first (40% rule)
    line_units: dict[bool, int] = field(default_factory=dict)  # cjk -> units per line

    @property
    def shortfall_lines(self) -> int:
        """Lines the plan could not cover (entry limits reached)."""
        return self.delta_lines - self.planned_lines

    def entry(self, path: str) -> EntryBudget | None:
        for item in self.entries:
            if item.path == path:
                return item
        return None

    def budgeted_paths(self) -> set[str]:
        """Description paths a ``rewrite_rows`` change may target."""
        return {item.path for item in self.entries if item.mode != "keep"}

    def apply_drops(self, data: dict[str, Any]) -> dict[str, Any]:
        """The resume without the bullets the plan drops first. The plan's paths and
        row counts refer to this version."""
        return drop_bullets(data, list(self.drops)) if self.drops else data

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_pages": self.target_pages,
            "status": self.status,
            "pages": self.pages,
            "last_page_fill": round(self.last_page_fill, 3),
            "text_format": self.text_format,
            "delta_lines": self.delta_lines,
            "planned_lines": self.planned_lines,
            "lines_per_page": round(self.lines_per_page, 1),
            "reliable": self.reliable,
            "drops": [bullet_path(*key) for key in self.drops],
            "entries": [item.to_dict() for item in self.entries],
        }


@dataclass
class _Work:
    """One work entry while the plan is built."""

    index: int
    label: str
    rows: list[str]
    styles: list[str]
    lines: list[int]
    scores: list[float]
    kept: list[int]  # source row index of each kept row
    cjk: bool
    line_units: int
    weight: float

    @property
    def total_lines(self) -> int:
        return sum(self.lines)

    @property
    def result_count(self) -> int:
        return sum(1 for style in self.styles if style == "result")


def _job_terms(job_keywords: dict[str, Any] | None) -> set[str]:
    terms: set[str] = set()
    for key in _JOB_TERM_KEYS:
        values = (job_keywords or {}).get(key)
        if isinstance(values, list):
            terms.update(
                v.strip().lower() for v in values if isinstance(v, str) and v.strip()
            )
    return terms


def _keyword_scores(
    data: dict[str, Any], job_keywords: dict[str, Any] | None
) -> dict[BulletKey, float]:
    """Fallback relevance: 20 points per distinct job term in a bullet, capped at 100."""
    terms = _job_terms(job_keywords)
    return {
        key: float(min(100, 20 * sum(1 for t in terms if keyword_in_text(t, text))))
        for key, text, _entry in iter_bullet_rows(data)
    }


def _end_year(years: Any) -> int | None:
    text = str(years or "")
    if _PRESENT_RE.search(text):
        return 9999
    found = [int(match.group(0)) for match in _YEAR_RE.finditer(text)]
    return max(found) if found else None


def _recency_ranks(entries: list[tuple[int, dict[str, Any]]]) -> dict[int, int]:
    """0 for the most recent entry. Uses the end year when entries carry one, else the
    list order (resumes list the latest role first)."""
    ends = {index: _end_year(entry.get("years")) for index, entry in entries}
    if not any(value is not None for value in ends.values()):
        return {index: rank for rank, (index, _entry) in enumerate(entries)}
    ordered = sorted(ends, key=lambda i: (-(ends[i] or 0), i))
    return {index: rank for rank, index in enumerate(ordered)}


def _entry_label(entry: dict[str, Any]) -> str:
    title = str(entry.get("title") or "").strip()
    company = str(entry.get("company") or "").strip()
    return " @ ".join(part for part in (title, company) if part) or "work entry"


def _row_style(styles: Any, index: int) -> str:
    if isinstance(styles, list) and index < len(styles):
        style = styles[index]
        if style in ("bullet", "result", "plain"):
            return str(style)
    return "bullet"


def _line_units_for(layout: LayoutMeasurement) -> dict[bool, int]:
    """Units that fill one planned line, for CJK and for other text."""
    result: dict[bool, int] = {}
    for cjk in (False, True):
        capacity = layout.capacity(cjk)
        if capacity is None:
            # No row wraps: the longest single-line row is a lower bound.
            singles = [
                row.units for row in layout.rows if row.cjk == cjk and row.lines == 1
            ]
            capacity = float(max(singles)) if singles else None
        if capacity is None or capacity <= 0:
            result[cjk] = DEFAULT_LINE_UNITS[cjk]
        else:
            result[cjk] = max(MIN_LINE_UNITS, int(capacity * LINE_SAFETY))
    return result


def _collect_work(
    data: dict[str, Any],
    layout: LayoutMeasurement,
    scores: dict[BulletKey, float],
    line_units: dict[bool, int],
    dropped: set[BulletKey],
) -> list[_Work]:
    entries = data.get("workExperience")
    if not isinstance(entries, list):
        return []
    candidates = [
        (index, entry)
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
        and isinstance(entry.get("description"), list)
        and not entry.get("condensed")
    ]
    ranks = _recency_ranks(candidates)
    measured = layout.rows_by_path()
    default_cjk = is_cjk_text(
        " ".join(str(row) for _i, e in candidates for row in e["description"])
    )
    work: list[_Work] = []
    for index, entry in candidates:
        description = entry["description"]
        path = f"workExperience[{index}].description"
        entry_cjk = is_cjk_text(" ".join(str(row) for row in description))
        entry_cjk = entry_cjk if description else default_cjk
        units = line_units[entry_cjk]
        rows: list[str] = []
        styles: list[str] = []
        lines: list[int] = []
        row_scores: list[float] = []
        kept: list[int] = []
        for j, raw in enumerate(description):
            if ("workExperience", index, j) in dropped:
                continue
            text = str(raw)
            found = measured.get(f"{path}[{j}]")
            if found is not None:
                row_lines = max(1, found.lines)
            else:
                row_lines = max(1, math.ceil(count_units(text, entry_cjk) / units))
            rows.append(text)
            styles.append(_row_style(entry.get("descriptionStyles"), j))
            lines.append(row_lines)
            row_scores.append(scores.get(("workExperience", index, j), 0.0))
            kept.append(j)
        relevance = sum(row_scores) / len(row_scores) if row_scores else 0.0
        weight = (20.0 + relevance) / (1.0 + 0.35 * ranks[index])
        work.append(
            _Work(
                index=index,
                label=_entry_label(entry),
                rows=rows,
                styles=styles,
                lines=lines,
                scores=row_scores,
                kept=kept,
                cjk=entry_cjk,
                line_units=units,
                weight=weight,
            )
        )
    return work


def _cut_capacity(work: _Work) -> int:
    """Most lines a condense can remove: every row shortened to one line, and at most
    CONDENSE_ROW_SHARE of the rows removed (never below MIN_ROWS)."""
    rows = len(work.lines)
    if rows == 0:
        return 0
    removable = min(int(rows * CONDENSE_ROW_SHARE + 1e-9), rows - MIN_ROWS)
    return max(0, work.total_lines - (rows - max(0, removable)))


def _allocate_cut(works: list[_Work], need: int) -> tuple[dict[int, int], int]:
    """Spread ``need`` lines over entries, least relevant and oldest first, one line at
    a time (largest ``lines / weight / (taken + 1)`` first). Returns (lines per entry,
    lines left uncovered)."""
    caps = {w.index: _cut_capacity(w) for w in works}
    taken = {w.index: 0 for w in works}
    remaining = need
    while remaining > 0:
        best: _Work | None = None
        best_key: tuple[float, int] | None = None
        for work in works:
            if taken[work.index] >= caps[work.index]:
                continue
            key = (work.total_lines / work.weight / (taken[work.index] + 1), work.index)
            if best_key is None or key > best_key:
                best, best_key = work, key
        if best is None:
            break
        taken[best.index] += 1
        remaining -= 1
    return taken, remaining


def _allocate_add(
    works: list[_Work], need: int
) -> tuple[dict[int, tuple[int, int]], int]:
    """Spread ``need`` extra lines over entries, most relevant and recent first.

    First as new rows (at most max(3, 50% of the entry's rows) each), then by
    lengthening one-line action rows to two lines. Returns ({entry: (new rows, extra
    lines)}, lines left uncovered)."""
    added: dict[int, list[int]] = {w.index: [0, 0] for w in works}  # rows, lines
    remaining = need

    def row_cost(work: _Work) -> int:
        if not work.lines:
            return ACTION_MAX_LINES
        mean = work.total_lines / len(work.lines)
        return max(RESULT_MAX_LINES, min(ACTION_MAX_LINES, round(mean)))

    while remaining > 0:
        best: _Work | None = None
        best_key: tuple[float, int] | None = None
        for work in works:
            cap = max(EXPAND_MIN_ROWS, math.ceil(len(work.lines) * EXPAND_SHARE))
            if added[work.index][0] >= cap:
                continue
            key = (work.weight / (added[work.index][0] + 1), -work.index)
            if best_key is None or key > best_key:
                best, best_key = work, key
        if best is None:
            break
        lines = min(row_cost(best), remaining)
        added[best.index][0] += 1
        added[best.index][1] += lines
        remaining -= lines

    slack = {
        w.index: sum(
            1
            for style, n in zip(w.styles, w.lines, strict=True)
            if style != "result" and n < ACTION_MAX_LINES
        )
        for w in works
    }
    grown = {w.index: 0 for w in works}
    while remaining > 0:
        best = None
        best_key = None
        for work in works:
            if grown[work.index] >= slack[work.index]:
                continue
            key = (work.weight / (grown[work.index] + 1), -work.index)
            if best_key is None or key > best_key:
                best, best_key = work, key
        if best is None:
            break
        grown[best.index] += 1
        remaining -= 1
    result = {
        w.index: (added[w.index][0], added[w.index][1] + grown[w.index]) for w in works
    }
    return result, remaining


def _wanted_result_rows(work: _Work, target_rows: int, text_format: bool) -> int:
    existing = work.result_count
    if not text_format or target_rows < 2:
        return existing
    return max(
        existing, min(target_rows - 1, max(1, round(target_rows * RESULT_ROW_SHARE)))
    )


def _entry_budget(
    work: _Work,
    *,
    mode: BudgetMode,
    target_rows: int,
    target_lines: int,
    text_format: bool,
    dropped: tuple[int, ...],
    result_rows: int | None = None,
) -> EntryBudget:
    units = work.line_units
    target_rows = max(MIN_ROWS, target_rows)
    if result_rows is None:
        result_rows = _wanted_result_rows(work, target_rows, text_format)
    return EntryBudget(
        path=f"workExperience[{work.index}].description",
        entry_index=work.index,
        label=work.label,
        mode=mode,
        current_rows=len(work.rows),
        current_lines=work.total_lines,
        target_rows=target_rows,
        target_lines=max(0, target_lines),
        min_rows=max(MIN_ROWS, target_rows - ROW_TOLERANCE),
        max_rows=target_rows + ROW_TOLERANCE,
        result_rows=result_rows,
        cjk=work.cjk,
        line_units=units,
        action_max_units=units * ACTION_MAX_LINES,
        result_max_units=units * RESULT_MAX_LINES,
        weight=round(work.weight, 3),
        dropped_rows=dropped,
    )


def _estimated_lines(
    key: BulletKey, text: str, layout: LayoutMeasurement, line_units: dict[bool, int]
) -> int:
    found = layout.rows_by_path().get(bullet_path(*key))
    if found is not None:
        return max(1, found.lines)
    cjk = is_cjk_text(text)
    return max(1, math.ceil(count_units(text, cjk) / line_units[cjk]))


def plan_budget(
    data: dict[str, Any],
    layout: LayoutMeasurement,
    target_pages: int | None,
    *,
    scores: dict[BulletKey, float] | None = None,
    job_keywords: dict[str, Any] | None = None,
    description_format: str = "bullet",
    adjust_lines: int = 0,
) -> ContentBudget:
    """Plan each work entry's length from one measured render.

    ``layout`` is a measurement of ``data`` rendered with the resume's own settings.
    ``scores`` is the bullet relevance (0 to 100); without it, job-keyword overlap is
    used. ``adjust_lines`` is what steps outside the rewrite will add (+) or remove (-)
    first, such as dropping skills.

    The target is a last page about ``TARGET_FILL`` full. A resume already inside the
    90% to 100% tolerance keeps its length. Over the target, entries are condensed;
    when that would cut more than 40% of an entry's rows, the least relevant bullets
    are dropped first (``ContentBudget.drops``, applied by ``apply_drops``). Under it,
    entries are expanded within ``EXPAND_MIN_ROWS`` / ``EXPAND_SHARE``. With no page
    target (unlimited) every entry keeps its current length.

    In the text format, entries without a result row are marked ``restructure`` (split
    sentences into an action row plus a result row) when nothing else changes them.
    """
    text_format = description_format == "text"
    scores = scores if scores is not None else _keyword_scores(data, job_keywords)
    reliable = (
        layout.probe_ok and layout.line_height_px > 0 and layout.page_height_px > 0
    )
    # Lines that fit on ONE page. ``content_height_px`` is the whole resume (it grows
    # with every page), so it cannot be used here.
    lines_per_page = (
        layout.page_height_px / layout.line_height_px
        if reliable
        else DEFAULT_LINES_PER_PAGE
    )
    line_units = _line_units_for(layout)

    if target_pages is None:
        status: BudgetStatus = "unlimited"
        need = 0
    else:
        status = classify_fill(layout.pages, layout.last_page_fill, target_pages)
        need = 0
        if status != "fit":
            gap = target_pages - layout.pages + TARGET_FILL - layout.last_page_fill
            need = round(gap * lines_per_page) - adjust_lines
            if status == "over":
                need = min(need, -1)
            elif status == "under":
                need = max(need, 1)

    works = _collect_work(data, layout, scores, line_units, set())
    drops: list[BulletKey] = []
    dropped_by_entry: dict[int, list[int]] = {}
    cut: dict[int, int] = {}
    added: dict[int, tuple[int, int]] = {}
    shortfall = 0

    if need < 0:
        order = trim_order(data, scores)
        dropped: set[BulletKey] = set()
        remaining_need = -need
        while True:
            works = _collect_work(data, layout, scores, line_units, dropped)
            cut, shortfall = _allocate_cut(works, remaining_need)
            if shortfall <= 0:
                break
            key = next((k for k in order if k not in dropped), None)
            if key is None:
                break
            row_text = next((t for k, t, _e in iter_bullet_rows(data) if k == key), "")
            dropped.add(key)
            drops.append(key)
            remaining_need = max(
                0, remaining_need - _estimated_lines(key, row_text, layout, line_units)
            )
            if key[0] == "workExperience":
                dropped_by_entry.setdefault(key[1], []).append(key[2])
        if shortfall > 0:
            logger.info(
                "Content budget: %d lines cannot be cut even after dropping %d bullets",
                shortfall,
                len(drops),
            )
        # The lines the dropped bullets free count toward the plan's coverage.
        freed = -need - remaining_need
        covered = freed + sum(cut.values())
        planned = -covered
    elif need > 0:
        added, shortfall = _allocate_add(works, need)
        planned = need - shortfall
    else:
        planned = 0

    entries: list[EntryBudget] = []
    for work in works:
        rows = len(work.rows)
        lines = work.total_lines
        dropped_rows = tuple(sorted(dropped_by_entry.get(work.index, [])))
        if need < 0 and cut.get(work.index, 0) > 0:
            taken = cut[work.index]
            shorten = sum(max(0, n - 1) for n in work.lines)
            removed = max(0, taken - shorten)
            entries.append(
                _entry_budget(
                    work,
                    mode="condense",
                    target_rows=rows - removed,
                    target_lines=lines - taken,
                    text_format=text_format,
                    dropped=dropped_rows,
                )
            )
        elif need > 0 and added.get(work.index, (0, 0))[1] > 0:
            new_rows, new_lines = added[work.index]
            entries.append(
                _entry_budget(
                    work,
                    mode="expand",
                    target_rows=rows + new_rows,
                    target_lines=lines + new_lines,
                    text_format=text_format,
                    dropped=dropped_rows,
                )
            )
        elif text_format and work.result_count == 0 and rows >= 1:
            wanted = max(1, round(rows * RESULT_ROW_SHARE))
            entries.append(
                _entry_budget(
                    work,
                    mode="restructure",
                    target_rows=rows + wanted,
                    target_lines=lines,
                    text_format=text_format,
                    dropped=dropped_rows,
                    result_rows=wanted,
                )
            )
        else:
            entries.append(
                _entry_budget(
                    work,
                    mode="keep",
                    target_rows=rows,
                    target_lines=lines,
                    text_format=text_format,
                    dropped=dropped_rows,
                )
            )

    return ContentBudget(
        target_pages=target_pages,
        status=status,
        pages=layout.pages,
        last_page_fill=layout.last_page_fill,
        text_format=text_format,
        delta_lines=need,
        planned_lines=planned,
        lines_per_page=lines_per_page,
        reliable=reliable,
        entries=tuple(entries),
        drops=tuple(drops),
        line_units=line_units,
    )


def budget_suggestions(budget: ContentBudget) -> list[str]:
    """Short plain-text next steps for an agent that writes the draft itself."""
    notes: list[str] = []
    if budget.status == "unlimited":
        notes.append("No page target: keep each entry at its current length.")
    elif budget.status == "fit":
        notes.append("The resume already fills the target. Keep its length.")
    elif budget.status == "over":
        notes.append(
            f"Over the target: remove about {-budget.delta_lines} lines. Cut Skills the "
            "job does not ask for first, then Awards, then Certifications, then "
            "condense work entries."
        )
    else:
        notes.append(
            f"Under the target: add about {budget.delta_lines} lines to work "
            "entries, using only facts and methods the master entry supports."
        )
    if budget.drops:
        notes.append(
            "Drop these least relevant bullets first: "
            + ", ".join(bullet_path(*key) for key in budget.drops)
            + "."
        )
    if budget.shortfall_lines:
        notes.append(
            f"Entry limits leave about {abs(budget.shortfall_lines)} lines uncovered."
        )
    if not budget.reliable:
        notes.append("Line counts are estimates: the layout probe did not answer.")
    return notes


def _mode_line(entry: EntryBudget, text_format: bool) -> str:
    unit = entry.unit
    parts = [
        f"{entry.path} | {entry.label} | {entry.mode.upper()}",
        f"now {entry.current_rows} rows, {entry.current_lines} lines",
    ]
    if entry.mode != "keep":
        parts.append(
            f"target {entry.target_rows} rows ({entry.min_rows} to {entry.max_rows}),"
            f" about {entry.target_lines} lines"
        )
    parts.append(
        f"action row at most {entry.action_max_units} {unit}"
        + (
            f", result row at most {entry.result_max_units} {unit}"
            if text_format
            else ""
        )
    )
    if text_format and entry.result_rows:
        parts.append(f"{entry.result_rows} result row(s)")
    if entry.dropped_rows:
        parts.append("least relevant rows already removed")
    return " | ".join(parts)


def render_budget_block(
    budget: ContentBudget, *, create_mode: bool = False, agent_mode: bool = False
) -> str:
    """The budget as prompt text. Empty when there is nothing to say (no page target
    and nothing to restructure).

    ``agent_mode`` words the "How to apply it" line for a calling agent that writes the
    whole resume data itself (the MCP ``plan_content_budget`` tool), not diffs."""
    active = [e for e in budget.entries if e.mode != "keep"]
    if budget.status == "unlimited" and not active:
        return ""
    lines: list[str] = ["CONTENT BUDGET (measured on this resume's real layout):"]
    if budget.target_pages is not None:
        state = (
            f"Page target: {budget.target_pages} page(s). Now: {budget.pages} page(s), "
            f"last page {round(budget.last_page_fill * 100)}% full."
        )
        if budget.status == "over":
            state += f" Remove about {-budget.delta_lines} lines in total."
        elif budget.status == "under":
            state += f" Add about {budget.delta_lines} lines in total."
        else:
            state += (
                f" Length is on target ({round(FILL_MIN * 100)}% to 100% counts as "
                "full): keep each entry's length."
            )
        lines.append(state)
    else:
        lines.append("No page target: keep each entry's current length.")
    if not budget.reliable:
        lines.append("(Line counts are estimates.)")
    lines.append(
        "Measure rows in characters for Chinese, Japanese and Korean text and in "
        "words for every other language. One line holds about "
        f"{budget.line_units.get(False, DEFAULT_LINE_UNITS[False])} words or "
        f"{budget.line_units.get(True, DEFAULT_LINE_UNITS[True])} characters. "
        f"An action row may take at most {ACTION_MAX_LINES} lines. "
        + (
            'A result row ("-> ...") takes at most '
            f"{RESULT_MAX_LINES} line and holds only the outcome of the action row "
            "above it."
            if budget.text_format
            else ""
        )
    )
    lines.append("Work entries:")
    for entry in budget.entries:
        lines.append("- " + _mode_line(entry, budget.text_format))
    if agent_mode:
        lines.append(
            "How to apply it: in resume_data, write every work entry's "
            '"description" list with the row count the work history above gives '
            "for it (plus or minus one; action rows and result rows count "
            "together), and keep descriptionStyles the same length as description"
            + (
                ', with "result" for a result row (or start the row with "-> ").'
                if budget.text_format
                else "."
            )
            + " Leave KEEP entries as they are."
        )
    elif create_mode:
        lines.append(
            'How to apply it: write every work entry\'s "description" list with the '
            "row count the work history above gives for it (plus or minus one; action "
            "rows and result rows count together)"
            + (', a result row starting with "-> ".' if budget.text_format else ".")
        )
    else:
        lines.append(
            "How to apply it: for every entry that is not KEEP, send ONE change with "
            'action "rewrite_rows" on that entry\'s description path. Its value is the '
            "complete list of rows for the entry, in order"
            + (', result rows starting with "-> ".' if budget.text_format else ".")
            + " Leave KEEP entries and every other field alone."
        )
    modes = {e.mode for e in active}
    if "condense" in modes:
        lines.append(
            "CONDENSE: tighten wording and merge or remove the least relevant rows. "
            "Keep every number and the strongest, most job-relevant rows; keep at "
            "least one row."
        )
    if "expand" in modes:
        if create_mode:
            lines.append(
                "EXPAND: add rows under the entry's existing company. Plausible "
                "numbers are allowed. Never copy names, companies or numbers from the "
                "style sample. Company, location and years never change."
            )
        else:
            lines.append(
                "EXPAND: add rows describing responsibilities and methods that fit "
                "this entry, combining the job description with what the entry "
                "already says. You may use tools and skills named in the job "
                "description. Use only numbers that already appear in this entry. "
                "Company, title, location and dates never change. Do not add the "
                "tools you mention to Skills."
            )
    if "restructure" in modes or (budget.text_format and (active or create_mode)):
        lines.append(
            "RESTRUCTURE: where a row holds an action and its outcome, split it into "
            'an action row and a result row that starts with "-> ". Do not repeat the '
            "action in the result row."
        )
    lines.append("Never use em dashes.")
    return "\n".join(lines)


def render_style_caps(budget: ContentBudget) -> str:
    """Per-row length caps for the writing loop's style rule. The loop only rewrites
    existing rows, so only the caps matter, not the row counts."""
    words = budget.line_units.get(False, DEFAULT_LINE_UNITS[False])
    chars = budget.line_units.get(True, DEFAULT_LINE_UNITS[True])
    text = (
        f"Length caps: one line holds about {words} words or {chars} characters. "
        f"An action row stays within {ACTION_MAX_LINES * words} words "
        f"({ACTION_MAX_LINES * chars} characters for Chinese, Japanese or Korean)."
    )
    if budget.text_format:
        text += (
            f" A result row stays within {RESULT_MAX_LINES * words} words "
            f"({RESULT_MAX_LINES * chars} characters)."
        )
    return text
