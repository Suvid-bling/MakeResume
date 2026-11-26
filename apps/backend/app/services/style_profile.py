"""Parse a style sample's text into structural writing-style metrics.

The "create" tailor mode asks the LLM to write new resume content that reads
like a user-chosen style sample, without copying its facts. This module turns
the sample's plain text into a ``StyleProfile``: countable, code-checkable
metrics (bullet counts, quantification ratio, bullet length range, whether
skills are grouped, protected proper nouns, distinctive numbers, and the raw
bullets themselves for overlap detection). ``app/services/resume_creator.py``
uses the profile to build the generation prompt and to check the LLM's output.

Parsing is a small line-based state machine, not a general resume parser: it
only has to understand the layout documented in ``app/prompts/style.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.prompts.style import StyleSample

_SECTION_HEADERS: dict[str, str] = {
    "个人简介": "summary",
    "summary": "summary",
    "profile": "summary",
    "工作经历": "experience",
    "experience": "experience",
    "work experience": "experience",
    "技能": "skills",
    "skills": "skills",
    "technical skills": "skills",
    "教育背景": "education",
    "education": "education",
}

# A "Label: a, b" or "Label：a、b" skills row (same shape the frontend and the
# backend ResumeData schema treat as a grouped skills line).
_SKILL_LABEL_RE = re.compile(r"^([^,:：，、]{1,40})(?::(?:\s+(.*))?|：\s*(.*))$")

# Numbers worth protecting against reuse: decimals, thousands separators,
# percentages, or any run of 2+ digits followed by a unit letter/word.
_DISTINCTIVE_NUMBER_RE = re.compile(
    r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*%|\d{2,}(?=\s*(?:x|s\b))"
)
_ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# CJK terminators always end a sentence. An ASCII ".", "!" or "?" only ends one
# when it is not preceded by a digit (decimals: "4.1") and is followed by
# whitespace or the end of the string (so "Node.js" does not split).
_SENTENCE_SPLIT_RE = re.compile(r"[。！？]+|(?<!\d)[.!?]+(?=\s|$)")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class ParsedEntry:
    title: str
    company: str
    location: str
    years: str
    bullets: tuple[str, ...]


@dataclass(frozen=True)
class ParsedSample:
    """Raw structural pieces pulled out of a style sample's text."""

    summary: str
    entries: tuple[ParsedEntry, ...]
    skills_lines: tuple[str, ...]
    protected_terms: frozenset[str]


@dataclass(frozen=True)
class StyleProfile:
    """Countable writing-style metrics derived from a :class:`ParsedSample`."""

    sample_id: str
    language: str
    summary_sentences: int
    bullets_per_entry: tuple[int, ...]  # most-recent-first, one count per entry
    quantified_ratio: float  # fraction of bullets containing at least one digit
    bullet_length_min: int
    bullet_length_max: int
    skills_grouped: bool
    skills_separator: str  # "、" or ", "
    protected_terms: frozenset[str]
    distinctive_numbers: frozenset[str]
    sample_bullets: tuple[str, ...]


def _is_date_line(line: str) -> bool:
    return bool(re.search(r"\d{4}|至今|至|present|-", line, re.IGNORECASE))


def _join_continuation(accumulated: str, continuation: str) -> str:
    """Append a wrapped bullet continuation, spacing ASCII runs but not CJK."""
    if not accumulated:
        return continuation
    left_ascii = bool(accumulated) and _ASCII_ALNUM_RE.match(accumulated[-1])
    right_ascii = bool(continuation) and _ASCII_ALNUM_RE.match(continuation[0])
    sep = " " if (left_ascii and right_ascii) else ""
    return accumulated + sep + continuation


def _split_entry_header(line: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in re.split(r"[，,]", line) if p.strip()]
    title = parts[0] if len(parts) > 0 else ""
    company = parts[1] if len(parts) > 1 else ""
    location = parts[2] if len(parts) > 2 else ""
    return title, company, location


def parse_style_sample(text: str) -> ParsedSample:
    """Parse a style sample's text into sections, entries and skills rows."""
    raw_lines = text.split("\n")
    non_blank = [line for line in raw_lines if line.strip()]
    protected_terms: set[str] = set()
    if non_blank:
        # The first non-blank line is the candidate's name/header line.
        name_line = non_blank[0].strip()
        name_candidate = name_line.split("·")[0].strip()
        if name_candidate:
            protected_terms.add(name_candidate)

    section: str | None = None
    summary_lines: list[str] = []
    skills_lines: list[str] = []
    entries: list[ParsedEntry] = []

    # Experience sub-state.
    state = "expect_header"  # expect_header -> expect_date -> in_bullets
    cur_title = cur_company = cur_location = cur_years = ""
    cur_bullets: list[str] = []

    def flush_entry() -> None:
        if cur_title or cur_company or cur_bullets:
            entries.append(
                ParsedEntry(
                    title=cur_title,
                    company=cur_company,
                    location=cur_location,
                    years=cur_years,
                    bullets=tuple(cur_bullets),
                )
            )

    for raw_line in raw_lines:
        if not raw_line.strip():
            continue
        stripped = raw_line.strip()
        leading_spaces = len(raw_line) - len(raw_line.lstrip(" "))
        lowered = stripped.lower()

        header_section = _SECTION_HEADERS.get(lowered) or _SECTION_HEADERS.get(stripped)
        if header_section is not None and leading_spaces == 0:
            if section == "experience":
                flush_entry()
                cur_title = cur_company = cur_location = cur_years = ""
                cur_bullets = []
                state = "expect_header"
            section = header_section
            continue

        if section == "summary":
            summary_lines.append(stripped)
        elif section == "skills":
            skills_lines.append(stripped)
        elif section == "experience":
            is_bullet = stripped.startswith("- ")
            if leading_spaces > 0 and not is_bullet:
                # Continuation of the previous bullet.
                if cur_bullets:
                    cur_bullets[-1] = _join_continuation(cur_bullets[-1], stripped)
                continue
            if is_bullet:
                cur_bullets.append(stripped[2:].strip())
                state = "in_bullets"
                continue
            if state == "in_bullets":
                # A new zero-indent, non-bullet line ends the current entry.
                flush_entry()
                cur_title, cur_company, cur_location = _split_entry_header(stripped)
                if cur_company:
                    protected_terms.add(cur_company)
                cur_years = ""
                cur_bullets = []
                state = "expect_date"
            elif state == "expect_header":
                cur_title, cur_company, cur_location = _split_entry_header(stripped)
                if cur_company:
                    protected_terms.add(cur_company)
                state = "expect_date"
            elif state == "expect_date":
                cur_years = stripped
                state = "in_bullets"
        elif section == "education":
            # Pull out protected nouns (institution, degree) but keep no
            # structured data — education is copied from the master verbatim.
            for part in re.split(r"[，,]", stripped):
                part = part.strip()
                if part and not re.search(r"^\d{4}$", part):
                    protected_terms.add(part)

    if section == "experience":
        flush_entry()

    summary = " ".join(summary_lines).strip()
    return ParsedSample(
        summary=summary,
        entries=tuple(entries),
        skills_lines=tuple(skills_lines),
        protected_terms=frozenset(t for t in protected_terms if len(t) > 1),
    )


def _bullet_length(bullet: str, language: str) -> int:
    if language == "zh":
        return len(_CJK_RE.findall(bullet)) or len(bullet.replace(" ", ""))
    return len(_WORD_RE.findall(bullet))


def _skills_grouping(skills_lines: tuple[str, ...]) -> tuple[bool, str]:
    grouped_count = 0
    zh_sep = en_sep = 0
    for line in skills_lines:
        match = _SKILL_LABEL_RE.match(line)
        if match and (match.group(2) or match.group(3) or "").strip():
            grouped_count += 1
            body = match.group(2) or match.group(3) or ""
            if "、" in body:
                zh_sep += 1
            if "," in body:
                en_sep += 1
    grouped = grouped_count >= max(1, len(skills_lines) // 2) and grouped_count > 0
    separator = "、" if zh_sep >= en_sep else ", "
    return grouped, separator


def _count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return len(parts) or 1


@lru_cache(maxsize=32)
def _compute_profile_cached(sample_id: str, language: str, text: str) -> StyleProfile:
    parsed = parse_style_sample(text)
    all_bullets = [b for entry in parsed.entries for b in entry.bullets]
    bullets_per_entry = tuple(len(entry.bullets) for entry in parsed.entries)
    quantified = sum(1 for b in all_bullets if re.search(r"\d", b))
    quantified_ratio = (quantified / len(all_bullets)) if all_bullets else 0.0
    lengths = [_bullet_length(b, language) for b in all_bullets] or [0]
    skills_grouped, separator = _skills_grouping(parsed.skills_lines)

    distinctive_numbers: set[str] = set()
    for b in all_bullets:
        distinctive_numbers.update(m.strip() for m in _DISTINCTIVE_NUMBER_RE.findall(b))

    return StyleProfile(
        sample_id=sample_id,
        language=language,
        summary_sentences=_count_sentences(parsed.summary),
        bullets_per_entry=bullets_per_entry,
        quantified_ratio=quantified_ratio,
        bullet_length_min=min(lengths),
        bullet_length_max=max(lengths),
        skills_grouped=skills_grouped,
        skills_separator=separator,
        protected_terms=parsed.protected_terms,
        distinctive_numbers=frozenset(distinctive_numbers),
        sample_bullets=tuple(all_bullets),
    )


def compute_style_profile(sample: StyleSample) -> StyleProfile:
    """Style metrics for ``sample``, cached by sample id for the process lifetime."""
    return _compute_profile_cached(sample.id, sample.language, sample.text)


def bullet_targets(profile: StyleProfile, n_entries: int, cap: int | None) -> list[int]:
    """Target bullet count for each of ``n_entries`` master entries.

    Entry ``i`` (0 = most recent) maps onto the sample entry at the
    proportionally equivalent position, so a short master maps onto the
    sample's strongest entries and a long master still covers its full span.
    """
    if n_entries <= 0:
        return []
    m = len(profile.bullets_per_entry)
    if m == 0:
        return [min(1, cap) if cap else 1 for _ in range(n_entries)]
    targets: list[int] = []
    for i in range(n_entries):
        idx = 0 if n_entries == 1 else round(i * (m - 1) / (n_entries - 1))
        idx = max(0, min(m - 1, idx))
        target = profile.bullets_per_entry[idx]
        if cap is not None:
            target = min(target, cap)
        targets.append(max(1, target))
    return targets
