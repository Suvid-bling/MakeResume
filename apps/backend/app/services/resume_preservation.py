"""Deterministic preservation at the final AI-authored resume seam."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from app.schemas.models import (
    SKILL_ROW_BREAK,
    has_skill_row_layout,
    skills_end_in_open_label,
)

GROUNDING_REVIEW_CODE = "GROUNDING_REVIEW_REQUIRED"
_GROUNDING_REVIEW_THRESHOLD = 0.45

_ENTRY_FIELDS: dict[str, tuple[str, ...]] = {
    "workExperience": ("company", "title"),
    "education": ("institution", "degree"),
    "personalProjects": ("name", "role"),
    "customItems": ("title", "subtitle"),
}
_PROTECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "workExperience": ("id", "title", "company", "location", "years", "condensed"),
    "education": ("id", "institution", "degree", "years"),
    "personalProjects": ("id", "name", "role", "years", "github", "website"),
    "customItems": ("id", "title", "subtitle", "location", "years"),
}
# Yes/no entry flags: a missing key reads as false, so legacy data compares equal.
_FLAG_FIELDS = frozenset({"condensed"})
_TOKEN_RE = re.compile(r"[\w+#./-]+", re.UNICODE)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<currency>[$€£])?(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>thousand|million|billion|percent|times|ms|gb|mb|tb|[kmb%x]))?"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "using",
        "with",
    }
)


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _identity(entry: dict[str, Any], section: str) -> tuple[str, ...]:
    entry_id = entry.get("id")
    if isinstance(entry_id, int) and entry_id != 0:
        return ("id", str(entry_id))
    return (
        "fields",
        *(_normalized(entry.get(field)) for field in _ENTRY_FIELDS[section]),
    )


def _field_identity(entry: dict[str, Any], section: str) -> tuple[str, ...] | None:
    identity = tuple(_normalized(entry.get(field)) for field in _ENTRY_FIELDS[section])
    return identity if any(identity) else None


def _matching_source_index(
    candidate: dict[str, Any],
    source_entries: list[dict[str, Any]],
    available: set[int],
    section: str,
) -> int | None:
    candidate_id = candidate.get("id")
    if isinstance(candidate_id, int) and candidate_id != 0:
        for index in sorted(available):
            if source_entries[index].get("id") == candidate_id:
                return index

    candidate_identity = _field_identity(candidate, section)
    if candidate_identity is None:
        return None
    matches = [
        index
        for index in sorted(available)
        if _field_identity(source_entries[index], section) == candidate_identity
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    candidate_description = candidate.get("description")
    candidate_text = (
        " ".join(str(row) for row in candidate_description)
        if isinstance(candidate_description, list)
        else str(candidate_description or "")
    )
    return max(
        matches,
        key=lambda index: (
            _similarity(
                " ".join(
                    str(row) for row in source_entries[index].get("description", [])
                )
                if isinstance(source_entries[index].get("description"), list)
                else str(source_entries[index].get("description") or ""),
                candidate_text,
            ),
            -index,
        ),
    )


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


def _similarity(left: str, right: str) -> float:
    left_normalized = _normalized(left)
    right_normalized = _normalized(right)
    if left_normalized == right_normalized:
        return 1.0
    left_tokens = _tokens(left_normalized)
    right_tokens = _tokens(right_normalized)
    overlap = len(left_tokens & right_tokens) / max(
        1, max(len(left_tokens), len(right_tokens))
    )
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    if left_tokens < right_tokens or right_tokens < left_tokens:
        # Character similarity must not hide a material expansion or truncation
        # when every token on the shorter side happens to match.
        return overlap * sequence
    return max(overlap, sequence)


def _is_date_like_number(text: str, match: re.Match[str]) -> bool:
    """Return whether a bare four-digit number appears in date context."""
    prefix = text[max(0, match.start() - 24) : match.start()].casefold()
    suffix = text[match.end() : min(len(text), match.end() + 16)].casefold()
    month = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)"
    )
    return bool(
        re.search(rf"\b{month}\s*$", prefix)
        or (
            re.search(r"\b(?:in|since|during|from|until|through|between)\s*$", prefix)
            and re.match(r"^\s*(?:$|[,.;:)])", suffix)
        )
        or re.search(r"\b\d{4}\s*[-–—/]\s*$", prefix)
        or re.match(r"^\s*[-–—/]\s*(?:\d{4}|present|current)\b", suffix)
    )


def _novel_numbers(source: str, candidate: str) -> bool:
    def numeric_claims(text: str) -> Counter[str]:
        claims: Counter[str] = Counter()
        for match in _NUMBER_RE.finditer(text):
            raw_number = match.group("number").replace(",", "")
            unit = (match.group("unit") or "").casefold()
            prefix = text[max(0, match.start() - 16) : match.start()].casefold()
            if re.search(r"(?:python|node(?:\.js)?|java|version|v)\s*$", prefix):
                continue
            if not unit and not match.group("currency"):
                try:
                    integer = int(raw_number)
                except ValueError:
                    integer = 0
                if 1900 <= integer <= 2100 and _is_date_like_number(text, match):
                    continue
            try:
                value = Decimal(raw_number)
            except InvalidOperation:
                continue
            scale = {
                "k": 1_000,
                "thousand": 1_000,
                "m": 1_000_000,
                "million": 1_000_000,
                "b": 1_000_000_000,
                "billion": 1_000_000_000,
            }.get(unit, 1)
            value *= scale
            normalized_value = format(value.normalize(), "f")
            kind = (
                match.group("currency")
                or ("%" if unit in {"%", "percent"} else "")
                or ("x" if unit in {"x", "times"} else "")
                or (unit if unit in {"ms", "gb", "mb", "tb"} else "")
            )
            claims[f"{kind}:{normalized_value}"] += 1
        return claims

    source_numbers = numeric_claims(source)
    candidate_numbers = numeric_claims(candidate)
    return any(
        count > source_numbers[value] for value, count in candidate_numbers.items()
    )


_ARROW_PREFIX_RE = re.compile(r"^\s*(?:->|→|➜)")
_ROW_PREFIX_RE = re.compile(r"^\s*(?:->|→|➜|[-*•‣◦▪▫]+|\d+[.)])\s*")
# bullet (action) and result rows are interchangeable; plain stays plain.
_MARKED_STYLES = {"bullet", "result"}
_KNOWN_STYLES = {"bullet", "result", "plain"}


def _normalized_candidate_rows(
    value: Any,
    *,
    one_row_per_item: bool = False,
    styles: Any = None,
) -> tuple[list[str], list[str | None]]:
    """Mirror ResumeData's newline and bullet cleanup before preservation.

    With ``one_row_per_item`` a multi-line item is collapsed into a single row,
    so a rewrite of one harness-chosen bullet can never displace another.

    Returns the cleaned rows and, per row, the style the candidate asked for:
    "result" for a row written with an arrow prefix, else the candidate's own
    ``descriptionStyles`` entry for the first row of an item (when ``styles`` is
    a list aligned one-to-one with ``value``), else ``None``.
    """
    if not isinstance(value, list):
        return [], []
    aligned = styles if isinstance(styles, list) and len(styles) == len(value) else None
    rows: list[str] = []
    hints: list[str | None] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            continue
        item_style = aligned[item_index] if aligned is not None else None
        if item_style not in _KNOWN_STYLES:
            item_style = None
        lines: list[str] = []
        line_hints: list[str | None] = []
        for line_index, raw_line in enumerate(re.split(r"\r?\n+", item)):
            line = _ROW_PREFIX_RE.sub("", raw_line).strip()
            if line:
                lines.append(line)
                if _ARROW_PREFIX_RE.match(raw_line) and item_style != "plain":
                    line_hints.append("result")
                else:
                    line_hints.append(item_style if line_index == 0 else None)
        if one_row_per_item and lines:
            rows.append(" ".join(lines))
            hints.append(line_hints[0])
        else:
            rows.extend(lines)
            hints.extend(line_hints)
    return rows, hints


def _row_style(
    source_styles: list[Any],
    source_index: int,
    hint: str | None = None,
    *,
    trust_hint: bool = False,
) -> str:
    """Style of a kept row: the source's, except bullet/result may swap.

    With ``trust_hint`` (length-fit mode, where rows are rewritten wholesale and the
    best-matching source row is only a guess) the candidate's own style wins.
    """
    if trust_hint and hint in _KNOWN_STYLES:
        return str(hint)
    source = source_styles[source_index] if source_index < len(source_styles) else None
    if source not in _KNOWN_STYLES:
        source = "bullet"
    if source in _MARKED_STYLES and hint in _MARKED_STYLES:
        return str(hint)
    return str(source)


def _best_source_row(
    candidate: str,
    source_rows: list[str],
    available: set[int],
) -> tuple[int, float] | None:
    if not available:
        return None
    scored = [
        (_similarity(source_rows[index], candidate), index) for index in available
    ]
    score, index = max(scored, key=lambda item: (item[0], -item[1]))
    return index, score


def _merge_description_rows(
    source_entry: dict[str, Any],
    candidate_entry: dict[str, Any],
    *,
    allow_review_claims: bool,
    allow_appended_rows: bool,
    fixed_rows: bool = False,
    length_fit: bool = False,
) -> tuple[list[str], list[str]]:
    source_rows = source_entry.get("description")
    candidate_rows = candidate_entry.get("description")
    if not isinstance(source_rows, list):
        return [], []
    source_text = " ".join(str(row) for row in source_rows)
    candidate_list, candidate_hints = _normalized_candidate_rows(
        candidate_rows,
        one_row_per_item=fixed_rows,
        styles=candidate_entry.get("descriptionStyles"),
    )
    source_styles = source_entry.get("descriptionStyles")
    styles = source_styles if isinstance(source_styles, list) else []
    available = set(range(len(source_rows)))
    merged_rows: list[str] = []
    merged_styles: list[str] = []

    for candidate_index, raw_candidate in enumerate(candidate_list[: len(source_rows)]):
        if not isinstance(raw_candidate, str):
            continue
        match = _best_source_row(raw_candidate, source_rows, available)
        if match is None:
            break
        source_index, score = match
        candidate_row = raw_candidate
        requires_restore = (
            not candidate_row.strip()
            or _novel_numbers(source_text, candidate_row)
            or (not allow_review_claims and score < _GROUNDING_REVIEW_THRESHOLD)
        )
        if not candidate_row.strip() and candidate_index in available:
            source_index = candidate_index
        available.remove(source_index)
        source_row = str(source_rows[source_index])
        if requires_restore:
            candidate_row = source_row
        merged_rows.append(candidate_row)
        merged_styles.append(
            _row_style(
                styles,
                source_index,
                None if requires_restore else candidate_hints[candidate_index],
                trust_hint=length_fit,
            )
        )

    # Length-fit mode lets an entry shrink, so rows the candidate left out stay out.
    # An entry never ends up empty.
    if not length_fit or not merged_rows:
        for source_index in sorted(available):
            merged_rows.append(str(source_rows[source_index]))
            merged_styles.append(_row_style(styles, source_index))
    if allow_appended_rows:
        for appended_index, candidate_row in enumerate(
            candidate_list[len(source_rows) :], start=len(source_rows)
        ):
            match = _best_source_row(
                candidate_row, source_rows, set(range(len(source_rows)))
            )
            grounded = match is not None and match[1] >= _GROUNDING_REVIEW_THRESHOLD
            if _novel_numbers(source_text, candidate_row) or (
                not allow_review_claims and not grounded
            ):
                continue
            merged_rows.append(candidate_row)
            hint = candidate_hints[appended_index]
            if length_fit and hint in _KNOWN_STYLES:
                merged_styles.append(str(hint))
            else:
                merged_styles.append("result" if hint == "result" else "bullet")
    return merged_rows, merged_styles


def _description_contract_preserved(
    source_entry: dict[str, Any],
    candidate_entry: dict[str, Any],
    *,
    allow_appended_rows: bool = False,
    length_fit: bool = False,
) -> bool:
    source_rows = source_entry.get("description")
    candidate_rows = candidate_entry.get("description")
    if not isinstance(source_rows, list):
        return True
    if not isinstance(candidate_rows, list):
        return False
    if length_fit:
        # Fewer or more rows are fine, as long as a non-empty entry keeps a row.
        if source_rows and not candidate_rows:
            return False
    elif len(candidate_rows) < len(source_rows):
        return False
    elif not allow_appended_rows and len(candidate_rows) != len(source_rows):
        return False
    if "descriptionStyles" not in source_entry:
        return True
    source_styles = source_entry.get("descriptionStyles")
    candidate_styles = candidate_entry.get("descriptionStyles")
    if not isinstance(source_styles, list) or not isinstance(candidate_styles, list):
        return False
    if len(candidate_styles) != len(candidate_rows):
        return False
    if length_fit:
        # Rows are rewritten wholesale, so the style of a row is the author's call.
        return all(
            isinstance(row, str) and style in _KNOWN_STYLES
            for row, style in zip(candidate_rows, candidate_styles)
        )

    available = set(range(len(source_rows)))
    for candidate_index, candidate_row in enumerate(candidate_rows[: len(source_rows)]):
        if not isinstance(candidate_row, str):
            return False
        match = _best_source_row(candidate_row, source_rows, available)
        if match is None:
            return False
        source_index, _ = match
        available.remove(source_index)
        expected_style = (
            source_styles[source_index]
            if source_index < len(source_styles)
            and source_styles[source_index] in _KNOWN_STYLES
            else "bullet"
        )
        actual_style = candidate_styles[candidate_index]
        if expected_style in _MARKED_STYLES:
            if actual_style not in _MARKED_STYLES:
                return False
        elif actual_style != expected_style:
            return False
    return True


def _merge_entries(
    source_entries: Any,
    candidate_entries: Any,
    section: str,
    *,
    allow_review_claims: bool,
    allow_appended_rows: bool,
    fixed_rows: bool = False,
    length_fit: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(source_entries, list):
        return []
    source_dicts = [entry for entry in source_entries if isinstance(entry, dict)]
    if not source_dicts:
        return []
    result: list[dict[str, Any]] = []
    available = set(range(len(source_dicts)))
    candidates = candidate_entries if isinstance(candidate_entries, list) else []
    for candidate_entry in candidates:
        if not isinstance(candidate_entry, dict):
            continue
        source_index = _matching_source_index(
            candidate_entry,
            source_dicts,
            available,
            section,
        )
        if source_index is None:
            continue
        available.remove(source_index)
        source_entry = source_dicts[source_index]
        merged = copy.deepcopy(candidate_entry)
        for field in _PROTECTED_FIELDS[section]:
            if field in source_entry:
                merged[field] = copy.deepcopy(source_entry[field])
            else:
                merged.pop(field, None)
        if section != "education":
            has_style_metadata = (
                "descriptionStyles" in source_entry or "descriptionStyles" in merged
            )
            rows, styles = _merge_description_rows(
                source_entry,
                merged,
                allow_review_claims=allow_review_claims,
                allow_appended_rows=allow_appended_rows,
                fixed_rows=fixed_rows,
                length_fit=length_fit,
            )
            merged["description"] = rows
            if has_style_metadata:
                merged["descriptionStyles"] = styles
            else:
                merged.pop("descriptionStyles", None)
        else:
            source_description = source_entry.get("description")
            candidate_description = merged.get("description")
            if (
                not isinstance(source_description, str)
                or not isinstance(candidate_description, str)
                or not candidate_description.strip()
            ):
                merged["description"] = copy.deepcopy(source_description)
            elif isinstance(source_description, str) and _novel_numbers(
                source_description, candidate_description
            ):
                merged["description"] = source_description
            elif (
                isinstance(source_description, str)
                and not allow_review_claims
                and _similarity(source_description, candidate_description)
                < _GROUNDING_REVIEW_THRESHOLD
            ):
                merged["description"] = source_description
        result.append(merged)

    for source_index in sorted(available):
        result.append(copy.deepcopy(source_dicts[source_index]))
    return result


def _length_fit_items(
    field: str, source_items: list[str], candidate_items: list[str]
) -> list[str]:
    """One Additional field when the candidate may drop items: nothing is re-added."""
    remaining_source = Counter(_normalized(item) for item in source_items)
    if field != "technicalSkills":
        kept: list[str] = []
        for item in candidate_items:
            key = _normalized(item)
            if remaining_source[key] > 0:
                kept.append(item)
                remaining_source[key] -= 1
        return kept
    if not has_skill_row_layout(source_items):
        return list(candidate_items)
    remaining_candidate = Counter(_normalized(item) for item in candidate_items)
    present: list[str] = []
    for item in source_items:
        key = _normalized(item)
        if remaining_candidate[key] > 0:
            present.append(item)
            remaining_candidate[key] -= 1
    added: list[str] = []
    remaining_source = Counter(_normalized(item) for item in source_items)
    for item in candidate_items:
        key = _normalized(item)
        if remaining_source[key] > 0:
            remaining_source[key] -= 1
        else:
            added.append(item)
    if added and skills_end_in_open_label(present):
        added.insert(0, SKILL_ROW_BREAK)
    return [*present, *added]


def _merge_additional(
    source: Any, candidate: Any, *, length_fit: bool = False
) -> dict[str, list[str]]:
    """Keep the source's Additional items in the candidate.

    With ``length_fit`` the candidate may drop any item (to fit the page target). It
    still cannot add awards, certifications or languages, and skills keep the source's
    row order when the source laid them out in rows.
    """
    source_dict = source if isinstance(source, dict) else {}
    candidate_dict = candidate if isinstance(candidate, dict) else {}
    result: dict[str, list[str]] = {}
    for field in ("technicalSkills", "languages", "certificationsTraining", "awards"):
        source_items = [
            item for item in source_dict.get(field, []) if isinstance(item, str)
        ]
        candidate_items = [
            item for item in candidate_dict.get(field, []) if isinstance(item, str)
        ]
        if length_fit:
            result[field] = _length_fit_items(field, source_items, candidate_items)
            continue
        if field != "technicalSkills":
            remaining_source_items = Counter(_normalized(item) for item in source_items)
            preserved_candidate_items: list[str] = []
            for item in candidate_items:
                key = _normalized(item)
                if remaining_source_items[key] > 0:
                    preserved_candidate_items.append(item)
                    remaining_source_items[key] -= 1
            candidate_items = preserved_candidate_items
        elif has_skill_row_layout(source_items):
            # The user laid the skills out in rows ("---" breaks or bare "Label:"
            # lines). Keep their order so an AI rewrite cannot move skills between
            # rows; new skills go last, in a row of their own after a bare label.
            remaining_source_items = Counter(_normalized(item) for item in source_items)
            added_items: list[str] = []
            for item in candidate_items:
                key = _normalized(item)
                if remaining_source_items[key] > 0:
                    remaining_source_items[key] -= 1
                else:
                    added_items.append(item)
            if added_items and skills_end_in_open_label(source_items):
                added_items.insert(0, SKILL_ROW_BREAK)
            result[field] = [*source_items, *added_items]
            continue
        remaining_candidate_items = Counter(
            _normalized(item) for item in candidate_items
        )
        result[field] = list(candidate_items)
        for item in source_items:
            key = _normalized(item)
            if remaining_candidate_items[key] > 0:
                remaining_candidate_items[key] -= 1
            else:
                result[field].append(item)
    return result


def _merge_custom_sections(
    source: Any,
    candidate: Any,
    *,
    allow_review_claims: bool,
    allow_appended_rows: bool,
    length_fit: bool = False,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    candidate_dict = candidate if isinstance(candidate, dict) else {}
    result: dict[str, Any] = {}
    for key, source_section in source.items():
        if not isinstance(source_section, dict):
            result[key] = copy.deepcopy(source_section)
            continue
        candidate_section = candidate_dict.get(key)
        if not isinstance(candidate_section, dict):
            result[key] = copy.deepcopy(source_section)
            continue
        if source_section.get("sectionType") != "itemList":
            result[key] = copy.deepcopy(source_section)
            continue
        merged_section = copy.deepcopy(candidate_section)
        merged_section["sectionType"] = source_section.get("sectionType")
        merged_section["items"] = _merge_entries(
            source_section.get("items"),
            candidate_section.get("items"),
            "customItems",
            allow_review_claims=allow_review_claims,
            allow_appended_rows=allow_appended_rows,
            length_fit=length_fit,
        )
        result[key] = merged_section
    return result


def finalize_ai_resume(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_review_claims: bool = True,
    allow_appended_rows: bool = False,
    fixed_row_sections: tuple[str, ...] = (),
    length_fit: bool = False,
) -> dict[str, Any]:
    """Return a non-mutating AI result that preserves the source contract.

    Weakly grounded rewrites remain when ``allow_review_claims`` is true so a
    preview can present them for explicit confirmation. Definite new metrics,
    extra rows, missing sections and identity drift are always repaired.
    Entries in ``fixed_row_sections`` never keep appended rows, and a multi-line
    rewrite of one of their rows stays one row.

    ``length_fit`` is for a resume rewritten to fit a page target: the candidate may
    drop skills, awards, certifications and languages, and an entry may have fewer
    (or, with ``allow_appended_rows`` implied, more) rows. Numbers that are not in the
    source entry, new awards or certifications, personalInfo and company, title and
    date changes are still repaired.
    """
    allow_appended_rows = allow_appended_rows or length_fit
    result = copy.deepcopy(candidate) if isinstance(candidate, dict) else {}
    result["personalInfo"] = copy.deepcopy(source.get("personalInfo", {}))
    source_summary = source.get("summary")
    candidate_summary = result.get("summary")
    if isinstance(source_summary, str):
        if not isinstance(candidate_summary, str) or not candidate_summary.strip():
            result["summary"] = source_summary
        elif _novel_numbers(source_summary, candidate_summary) or (
            not allow_review_claims
            and _similarity(source_summary, candidate_summary)
            < _GROUNDING_REVIEW_THRESHOLD
        ):
            result["summary"] = source_summary

    for section in ("workExperience", "education", "personalProjects"):
        result[section] = _merge_entries(
            source.get(section),
            result.get(section),
            section,
            allow_review_claims=allow_review_claims,
            allow_appended_rows=allow_appended_rows
            and section not in fixed_row_sections,
            fixed_rows=section in fixed_row_sections,
            length_fit=length_fit,
        )
    result["additional"] = _merge_additional(
        source.get("additional"), result.get("additional"), length_fit=length_fit
    )
    result["customSections"] = _merge_custom_sections(
        source.get("customSections"),
        result.get("customSections"),
        allow_review_claims=allow_review_claims,
        allow_appended_rows=allow_appended_rows,
        length_fit=length_fit,
    )
    if "sectionMeta" in source:
        result["sectionMeta"] = copy.deepcopy(source["sectionMeta"])
    return result


def _entry_map(
    entries: Any, section: str
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    result: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                result[_identity(entry, section)].append(entry)
    return result


def _protected_field_changed(
    field: str, source_entry: dict[str, Any], candidate_entry: dict[str, Any]
) -> bool:
    if field in _FLAG_FIELDS:
        return bool(source_entry.get(field)) != bool(candidate_entry.get(field))
    return field in source_entry and source_entry.get(field) != candidate_entry.get(
        field
    )


def _entry_bucket_violation(
    source_entries: list[dict[str, Any]],
    candidate_entries: list[dict[str, Any]],
    section: str,
    *,
    allow_appended_rows: bool,
    length_fit: bool = False,
) -> str | None:
    """Validate one identity bucket using the finalizer's content matching."""
    available = set(range(len(source_entries)))
    for candidate_entry in candidate_entries:
        source_index = _matching_source_index(
            candidate_entry, source_entries, available, section
        )
        if source_index is None:
            return "identity"
        available.remove(source_index)
        source_entry = source_entries[source_index]
        if any(
            _protected_field_changed(field, source_entry, candidate_entry)
            for field in _PROTECTED_FIELDS[section]
        ):
            return "identity"
        if not _description_contract_preserved(
            source_entry,
            candidate_entry,
            allow_appended_rows=allow_appended_rows,
            length_fit=length_fit,
        ):
            return "descriptions"
    return None


def validate_confirmed_resume(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_appended_rows: bool = False,
    length_fit: bool = False,
) -> list[str]:
    """Return stable source-contract violation codes for a confirm payload.

    ``length_fit`` lets the candidate drop skills, awards, certifications and
    languages and keep fewer (or more) rows per entry. It still reports added awards,
    certifications or languages, changed entries and a work entry left without rows.
    """
    allow_appended_rows = allow_appended_rows or length_fit
    violations: list[str] = []
    if (
        isinstance(source.get("summary"), str)
        and source["summary"].strip()
        and not _normalized(candidate.get("summary"))
    ):
        violations.append("summary")

    for section in ("workExperience", "education", "personalProjects"):
        source_map = _entry_map(source.get(section), section)
        candidate_map = _entry_map(candidate.get(section), section)
        if Counter({key: len(value) for key, value in source_map.items()}) != Counter(
            {key: len(value) for key, value in candidate_map.items()}
        ):
            violations.append(f"{section}.entries")
            continue
        for key, source_entries in source_map.items():
            candidate_entries = candidate_map[key]
            violation = _entry_bucket_violation(
                source_entries,
                candidate_entries,
                section,
                allow_appended_rows=allow_appended_rows,
                length_fit=length_fit,
            )
            if violation is not None:
                violations.append(f"{section}.{violation}")

    source_custom = source.get("customSections")
    candidate_custom = candidate.get("customSections")
    if isinstance(source_custom, dict):
        candidate_custom_dict = (
            candidate_custom if isinstance(candidate_custom, dict) else {}
        )
        if set(source_custom) - set(candidate_custom_dict):
            violations.append("customSections")
        for key, source_section in source_custom.items():
            candidate_section = candidate_custom_dict.get(key)
            if not isinstance(source_section, dict) or not isinstance(
                candidate_section, dict
            ):
                continue
            if source_section.get("sectionType") != candidate_section.get(
                "sectionType"
            ):
                violations.append(f"customSections.{key}.sectionType")
                continue
            if source_section.get("sectionType") != "itemList":
                source_value = source_section.get("strings", source_section.get("text"))
                candidate_value = candidate_section.get(
                    "strings", candidate_section.get("text")
                )
                if source_value != candidate_value:
                    violations.append(f"customSections.{key}.content")
                continue
            source_map = _entry_map(source_section.get("items"), "customItems")
            candidate_map = _entry_map(candidate_section.get("items"), "customItems")
            if Counter(
                {identity: len(entries) for identity, entries in source_map.items()}
            ) != Counter(
                {identity: len(entries) for identity, entries in candidate_map.items()}
            ):
                violations.append(f"customSections.{key}.entries")
                continue
            for identity, source_entries in source_map.items():
                violation = _entry_bucket_violation(
                    source_entries,
                    candidate_map[identity],
                    "customItems",
                    allow_appended_rows=allow_appended_rows,
                    length_fit=length_fit,
                )
                if violation is not None:
                    violations.append(f"customSections.{key}.{violation}")

    source_additional = source.get("additional")
    candidate_additional = candidate.get("additional")
    if isinstance(source_additional, dict):
        candidate_dict = (
            candidate_additional if isinstance(candidate_additional, dict) else {}
        )
        for field in (
            "technicalSkills",
            "languages",
            "certificationsTraining",
            "awards",
        ):
            source_items = Counter(
                _normalized(item)
                for item in source_additional.get(field, [])
                if isinstance(item, str)
            )
            candidate_items = Counter(
                _normalized(item)
                for item in candidate_dict.get(field, [])
                if isinstance(item, str)
            )
            missing = source_items - candidate_items
            extras = candidate_items - source_items
            if (missing and not length_fit) or (field != "technicalSkills" and extras):
                violations.append(f"additional.{field}")
    return list(dict.fromkeys(violations))


def _grounding_warning(path: str) -> str:
    return f"{GROUNDING_REVIEW_CODE}: Review {path} against the source resume."


def _entry_grounding_warnings(
    source_entries: Any,
    candidate_entries: Any,
    section: str,
    path_prefix: str,
) -> list[str]:
    warnings: list[str] = []
    if not isinstance(source_entries, list) or not isinstance(candidate_entries, list):
        return warnings
    source_dicts = [entry for entry in source_entries if isinstance(entry, dict)]
    available_entries = set(range(len(source_dicts)))
    for candidate_index, candidate_entry in enumerate(candidate_entries):
        if not isinstance(candidate_entry, dict):
            continue
        source_index = _matching_source_index(
            candidate_entry, source_dicts, available_entries, section
        )
        if source_index is None:
            continue
        available_entries.remove(source_index)
        source_entry = source_dicts[source_index]
        source_description = source_entry.get("description")
        candidate_description = candidate_entry.get("description")
        description_path = f"{path_prefix}[{candidate_index}].description"
        if isinstance(source_description, str) and isinstance(
            candidate_description, str
        ):
            if (
                _normalized(source_description) != _normalized(candidate_description)
                and _similarity(source_description, candidate_description)
                < _GROUNDING_REVIEW_THRESHOLD
            ):
                warnings.append(_grounding_warning(description_path))
            continue
        if not isinstance(source_description, list) or not isinstance(
            candidate_description, list
        ):
            continue
        available_rows = set(range(len(source_description)))
        row_assignments: dict[int, tuple[int, float]] = {}
        for row_index, row in enumerate(candidate_description):
            if not isinstance(row, str):
                continue
            exact = next(
                (
                    source_index
                    for source_index in sorted(available_rows)
                    if _normalized(source_description[source_index]) == _normalized(row)
                ),
                None,
            )
            if exact is not None:
                row_assignments[row_index] = (exact, 1.0)
                available_rows.remove(exact)
        for row_index, row in enumerate(candidate_description):
            if not isinstance(row, str) or row_index in row_assignments:
                continue
            match = _best_source_row(
                row,
                source_description,
                available_rows,
            )
            if match is None:
                continue
            source_index, score = match
            available_rows.remove(source_index)
            row_assignments[row_index] = (source_index, score)
        for row_index, row in enumerate(candidate_description):
            if not isinstance(row, str):
                continue
            assignment = row_assignments.get(row_index)
            if assignment is None:
                match = _best_source_row(
                    row, source_description, set(range(len(source_description)))
                )
                if _normalized(row) and (
                    match is None or match[1] < _GROUNDING_REVIEW_THRESHOLD
                ):
                    warnings.append(
                        _grounding_warning(f"{description_path}[{row_index}]")
                    )
                continue
            source_index, score = assignment
            if (
                _normalized(row) != _normalized(source_description[source_index])
                and score < _GROUNDING_REVIEW_THRESHOLD
            ):
                warnings.append(_grounding_warning(f"{description_path}[{row_index}]"))
    return warnings


def grounding_review_warnings(
    source: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    """Return stable warnings for narrative rewrites with weak source overlap."""
    warnings: list[str] = []
    source_summary = source.get("summary")
    candidate_summary = candidate.get("summary")
    if (
        isinstance(source_summary, str)
        and isinstance(candidate_summary, str)
        and _normalized(source_summary) != _normalized(candidate_summary)
        and _similarity(source_summary, candidate_summary) < _GROUNDING_REVIEW_THRESHOLD
    ):
        warnings.append(_grounding_warning("summary"))

    for section in ("workExperience", "personalProjects", "education"):
        warnings.extend(
            _entry_grounding_warnings(
                source.get(section),
                candidate.get(section),
                section,
                section,
            )
        )

    source_custom = source.get("customSections")
    candidate_custom = candidate.get("customSections")
    if isinstance(source_custom, dict) and isinstance(candidate_custom, dict):
        for key, source_section in source_custom.items():
            candidate_section = candidate_custom.get(key)
            if (
                not isinstance(source_section, dict)
                or not isinstance(candidate_section, dict)
                or source_section.get("sectionType") != "itemList"
            ):
                continue
            warnings.extend(
                _entry_grounding_warnings(
                    source_section.get("items"),
                    candidate_section.get("items"),
                    "customItems",
                    f"customSections.{key}.items",
                )
            )
    return warnings
