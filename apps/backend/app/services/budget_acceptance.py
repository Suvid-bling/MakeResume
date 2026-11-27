"""Per-entry acceptance of ``rewrite_rows`` changes made against a content budget.

``rewrite_rows`` replaces ALL rows of one work entry in a single change (inserting rows
into the middle of an entry would break row-by-row matching). Each entry is checked on
its own and kept or reverted as a whole, so one bad entry never costs the others.
"""

import copy
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.services.content_budget import ContentBudget
from app.services.resume_preservation import (
    _ARROW_PREFIX_RE,
    _GROUNDING_REVIEW_THRESHOLD,
    _ROW_PREFIX_RE,
    _best_source_row,
    _matching_source_index,
    _normalized,
    _novel_numbers,
)
from app.services.writing_quality import scrub_em_dashes

logger = logging.getLogger(__name__)

REWRITE_ROWS_ACTION = "rewrite_rows"
_ENTRY_PATH_RE = re.compile(r"^workExperience\[(\d+)\]\.description$")

# Why an entry was reverted (also logged server-side; clients get no detail).
REASON_NOT_BUDGETED = "not_budgeted"
REASON_BAD_VALUE = "bad_value"
REASON_ROW_COUNT = "row_count"
REASON_NEW_NUMBER = "new_number"
REASON_LOST_NUMBER = "lost_number"
REASON_UNKNOWN_ENTRY = "unknown_entry"


@dataclass
class AcceptanceResult:
    """The source with every accepted entry merged in."""

    data: dict[str, Any]
    accepted: list[str] = field(default_factory=list)  # entry description paths
    reverted: dict[str, str] = field(default_factory=dict)  # path -> reason
    # Rows (``workExperience[i].description[j]``) weakly grounded in the master entry.
    review_paths: list[str] = field(default_factory=list)


def rewrites_from_changes(changes: Iterable[Any]) -> dict[str, Any]:
    """``{path: rows}`` of the ``rewrite_rows`` changes in a diff result. Works on
    ``ResumeChange`` objects and plain dicts; later changes to a path win."""
    rewrites: dict[str, Any] = {}
    for change in changes:
        if isinstance(change, Mapping):
            action, path, value = (
                change.get("action"),
                change.get("path"),
                change.get("value"),
            )
        else:
            action = getattr(change, "action", None)
            path = getattr(change, "path", None)
            value = getattr(change, "value", None)
        if action == REWRITE_ROWS_ACTION and isinstance(path, str):
            rewrites[path] = value
    return rewrites


def _clean_rows(value: Any, keep_plain: bool) -> tuple[list[str], list[str]] | None:
    """Rows and their styles from a ``rewrite_rows`` value, or None if it is malformed.

    A row starting with ``->`` (or an arrow) is a result row; every other row is an
    action row, or plain when the entry was plain. Markers are stripped, em dashes
    replaced and empty rows dropped.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    rows: list[str] = []
    styles: list[str] = []
    for item in value:
        for raw in re.split(r"\r?\n+", item):
            text = scrub_em_dashes(_ROW_PREFIX_RE.sub("", raw).strip())
            if not text:
                continue
            rows.append(text)
            if _ARROW_PREFIX_RE.match(raw):
                styles.append("result")
            else:
                styles.append("plain" if keep_plain else "bullet")
    return rows, styles


def _entry_rows(entry: dict[str, Any]) -> list[str]:
    description = entry.get("description")
    return (
        [row for row in description if isinstance(row, str)]
        if isinstance(description, list)
        else []
    )


def _all_plain(entry: dict[str, Any]) -> bool:
    styles = entry.get("descriptionStyles")
    rows = _entry_rows(entry)
    return (
        bool(rows)
        and isinstance(styles, list)
        and all(i < len(styles) and styles[i] == "plain" for i in range(len(rows)))
    )


def _master_rows(
    source_entry: dict[str, Any], master: dict[str, Any] | None
) -> list[str]:
    """Rows of the master entry that matches ``source_entry`` plus the source's own
    rows (a bullet kept from the master may have been rewritten already)."""
    rows = _entry_rows(source_entry)
    entries = master.get("workExperience") if isinstance(master, dict) else None
    if not isinstance(entries, list):
        return rows
    dicts = [entry for entry in entries if isinstance(entry, dict)]
    index = _matching_source_index(
        source_entry, dicts, set(range(len(dicts))), "workExperience"
    )
    if index is None:
        return rows
    return [*_entry_rows(dicts[index]), *rows]


def _weakly_grounded(row: str, reference: list[str]) -> bool:
    if not reference:
        return True
    if any(_normalized(row) == _normalized(item) for item in reference):
        return False
    match = _best_source_row(row, reference, set(range(len(reference))))
    return match is None or match[1] < _GROUNDING_REVIEW_THRESHOLD


def accept_rewrite_rows(
    source: dict[str, Any],
    rewrites: Mapping[str, Any],
    budget: ContentBudget,
    *,
    master: dict[str, Any] | None = None,
    create_mode: bool = False,
) -> AcceptanceResult:
    """Merge the ``rewrite_rows`` entries that pass their checks into a copy of ``source``.

    ``rewrites`` is ``{"workExperience[i].description": [rows]}`` (see
    ``rewrites_from_changes``). ``source`` is the data the budget was planned on, after
    ``budget.apply_drops``. An entry is reverted (left as in ``source``) when it

    * is not budgeted (mode keep, unknown path, a condensed entry),
    * is not a list of strings,
    * has a row count outside the budget's target +-1 (action and result rows together),
    * in normal Tailor (``create_mode`` False), has a number that appears nowhere in the
      master entry (create mode allows plausible numbers),
    * is a condense that loses a number the entry had.

    Accepted rows are stripped of list markers and em dashes; rows with an arrow prefix
    become result rows. In normal Tailor, rows of an expanded entry whose best
    similarity to any master row is below the grounding threshold (0.45) are listed in
    ``review_paths`` for the user to check. Nothing else of an entry changes.
    """
    result = AcceptanceResult(data=copy.deepcopy(source))
    entries = result.data.get("workExperience")
    for path, value in rewrites.items():
        match = _ENTRY_PATH_RE.match(path) if isinstance(path, str) else None
        if match is None or not isinstance(entries, list):
            result.reverted[str(path)] = REASON_UNKNOWN_ENTRY
            continue
        index = int(match.group(1))
        entry = entries[index] if index < len(entries) else None
        if not isinstance(entry, dict) or entry.get("condensed"):
            result.reverted[path] = REASON_UNKNOWN_ENTRY
            continue
        plan = budget.entry(path)
        if plan is None or plan.mode == "keep":
            result.reverted[path] = REASON_NOT_BUDGETED
            continue
        cleaned = _clean_rows(value, keep_plain=_all_plain(entry))
        if cleaned is None or not cleaned[0]:
            result.reverted[path] = REASON_BAD_VALUE
            continue
        rows, styles = cleaned
        if not plan.min_rows <= len(rows) <= plan.max_rows:
            result.reverted[path] = REASON_ROW_COUNT
            continue
        source_rows = _entry_rows(source["workExperience"][index])
        if rows == source_rows:
            continue  # nothing to merge
        reference = _master_rows(source["workExperience"][index], master)
        if not create_mode:
            reference_text = "\n".join(reference)
            if any(_novel_numbers(reference_text, row) for row in rows):
                result.reverted[path] = REASON_NEW_NUMBER
                continue
        if plan.mode == "condense" and _novel_numbers(
            "\n".join(rows), "\n".join(source_rows)
        ):
            result.reverted[path] = REASON_LOST_NUMBER
            continue
        entry["description"] = rows
        entry["descriptionStyles"] = styles
        result.accepted.append(path)
        if not create_mode and plan.mode == "expand":
            result.review_paths.extend(
                f"{path}[{j}]"
                for j, row in enumerate(rows)
                if _weakly_grounded(row, reference)
            )
    for path, reason in result.reverted.items():
        logger.info("rewrite_rows reverted: path=%s reason=%s", path, reason)
    return result
