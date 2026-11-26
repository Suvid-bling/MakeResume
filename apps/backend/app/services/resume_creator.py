"""Generate "create" tailor mode content: new resume text written from a job
description, styled after a user-chosen sample, merged onto an unmodified
master resume skeleton.

Unlike the other tailor modes, this one does not go through
``app.services.improver.apply_diffs`` or ``refine_resume`` — those enforce
truthfulness against the master, which "create" explicitly does not follow
for its generated fields (title, summary, work titles/bullets, skills).
Everything else (company, location, years, entry count, education, projects,
languages, certifications, awards, customSections, sectionMeta) is copied
from the master by :func:`merge_created_content`, never touched by the LLM.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

from app.llm import complete_json
from app.prompts import (
    CREATE_RESUME_PROMPT,
    CREATE_RESUME_REPAIR_PROMPT,
    get_language_name,
)
from app.schemas.models import CreateCheckItem, ResumeData
from app.services.improver import (
    _prepare_keywords_for_prompt,
    _resolve_path,
    _sanitize_user_input,
    _set_at_path,
)
from app.services.style_profile import (
    _SKILL_LABEL_RE,
    StyleProfile,
    _bullet_length,
    bullet_targets,
)
from app.services.writing_quality import scrub_em_dashes

logger = logging.getLogger(__name__)

# Overlap thresholds for "did the LLM copy the sample" detection.
_ZH_OVERLAP_CHARS = 12
_EN_OVERLAP_WORDS = 8
_EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_BULLET_PATH_RE = re.compile(r"\.description\[\d+\]$")


# ---------------------------------------------------------------------------
# Work skeleton: what the LLM is told about the master's work entries
# ---------------------------------------------------------------------------


def build_work_skeleton(master_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Index/company/location/years for every master work entry, in order."""
    entries = master_data.get("workExperience") or []
    skeleton: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        skeleton.append(
            {
                "index": index,
                "company": str(entry.get("company") or "").strip(),
                "location": str(entry.get("location") or "").strip(),
                "years": str(entry.get("years") or "").strip(),
            }
        )
    return skeleton


def render_work_skeleton(skeleton: list[dict[str, Any]], targets: list[int]) -> str:
    """Readable "index | company | location | years | target bullets" lines."""
    if not skeleton:
        return "(no work experience entries — write only title, summary and skills)"
    lines = []
    for entry, target in zip(skeleton, targets):
        parts = [
            str(entry["index"]),
            entry["company"] or "(unnamed company)",
            entry["location"] or "",
            entry["years"] or "",
            f"target bullets: {target}",
        ]
        lines.append(" | ".join(p for p in parts if p))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Style rules rendered into the prompt
# ---------------------------------------------------------------------------


def render_style_rules(
    profile: StyleProfile, targets: list[int], *, same_language: bool
) -> str:
    """Render ``profile`` into a numbered rule list for the generation prompt."""
    rules: list[str] = []
    if targets:
        rules.append("- Match the target bullet count given for each entry below (±1).")
    if profile.quantified_ratio >= 0.5:
        rules.append(
            "- Nearly every bullet should state a quantified result (a number, "
            "percentage, count or duration), as in the reference sample."
        )
    if profile.skills_grouped:
        label_style = (
            "Label: a, b" if profile.skills_separator == ", " else "Label：a、b"
        )
        rules.append(
            f'- Group the skills into labeled rows like "{label_style}", one row '
            "per category, as the reference sample does."
        )
    rules.append(
        f"- Write the summary as {profile.summary_sentences} sentence"
        f"{'s' if profile.summary_sentences != 1 else ''}, like the reference sample."
    )
    if same_language:
        unit = "characters" if profile.language == "zh" else "words"
        rules.append(
            f"- Keep each bullet roughly {profile.bullet_length_min} to "
            f"{profile.bullet_length_max} {unit} long, like the reference sample."
        )
        rules.append(
            "- Match the reference sample's sentence structure: action, object "
            "(with tools/technologies named), then result."
        )
    return "\n".join(rules)


def render_writing_style_rule(profile: StyleProfile, *, same_language: bool) -> str:
    """A single extra rule for the writing-loop rewrite prompt's mode_rule slot."""
    if not same_language:
        return ""
    unit = "characters" if profile.language == "zh" else "words"
    return (
        "6. Style: keep each rewritten line about "
        f"{profile.bullet_length_min} to {profile.bullet_length_max} {unit} "
        "long, matching the reference style sample this resume was generated "
        "from, as long as rules 2 and 3 hold.\n"
    )


def revert_style_drift(
    before: dict[str, Any],
    after: dict[str, Any],
    rewritten_paths: list[str],
    profile: StyleProfile,
) -> dict[str, Any]:
    """Undo a writing-loop rewrite that pushed a bullet out of the style's length range.

    Only considers bullet paths (``...description[i]``) that the writing loop
    actually rewrote. A bullet that was already outside the range before the
    rewrite is left as the loop produced it — there was nothing to preserve.
    """
    result = copy.deepcopy(after)
    low, high = profile.bullet_length_min, profile.bullet_length_max
    for path in rewritten_paths:
        if not _BULLET_PATH_RE.search(path):
            continue
        before_val, before_ok = _resolve_path(before, path)
        after_val, after_ok = _resolve_path(result, path)
        if not (before_ok and after_ok):
            continue
        if not (isinstance(before_val, str) and isinstance(after_val, str)):
            continue
        before_in_range = low <= _bullet_length(before_val, profile.language) <= high
        after_in_range = low <= _bullet_length(after_val, profile.language) <= high
        if before_in_range and not after_in_range:
            _set_at_path(result, path, before_val)
    return result


# ---------------------------------------------------------------------------
# Shape validation (response_validator for complete_json)
# ---------------------------------------------------------------------------


def _validate_created_shape(
    result: dict[str, Any], expected_indices: set[int]
) -> dict[str, Any]:
    """Validate the LLM's create-mode output shape. Raises ValueError on failure."""
    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("create response requires a non-empty 'title'")
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("create response requires a non-empty 'summary'")

    work = result.get("workExperience")
    if not isinstance(work, list):
        raise ValueError("create response requires a 'workExperience' list")
    seen_indices: set[int] = set()
    for item in work:
        if not isinstance(item, dict):
            raise ValueError("each workExperience item must be an object")
        index = item.get("index")
        if not isinstance(index, int) or index not in expected_indices:
            raise ValueError(f"workExperience item has an unexpected index: {index!r}")
        if index in seen_indices:
            raise ValueError(f"workExperience index {index} repeated")
        seen_indices.add(index)
        item_title = item.get("title")
        if not isinstance(item_title, str) or not item_title.strip():
            raise ValueError(f"workExperience[{index}] requires a non-empty title")
        description = item.get("description")
        if (
            not isinstance(description, list)
            or not description
            or not all(isinstance(d, str) and d.strip() for d in description)
        ):
            raise ValueError(
                f"workExperience[{index}] requires at least one non-empty bullet"
            )
    if seen_indices != expected_indices:
        raise ValueError(
            f"workExperience must cover indices {sorted(expected_indices)}, "
            f"got {sorted(seen_indices)}"
        )

    skills = result.get("technicalSkills")
    if (
        not isinstance(skills, list)
        or not skills
        or not all(isinstance(s, str) and s.strip() for s in skills)
    ):
        raise ValueError("create response requires a non-empty 'technicalSkills' list")

    return result


# ---------------------------------------------------------------------------
# Merge: everything the master owns, plus the LLM's generated fields
# ---------------------------------------------------------------------------


def merge_created_content(
    master_data: dict[str, Any],
    content: dict[str, Any],
    *,
    role_title: str,
    max_bullets_per_entry: int | None,
) -> dict[str, Any]:
    """Merge generated ``content`` onto an unmodified copy of ``master_data``.

    Only personalInfo.title, summary, each work entry's title/description and
    additional.technicalSkills change. Everything else — company, location,
    years, entry count, education, projects, languages, certifications,
    awards, customSections, sectionMeta — is the master's, untouched.
    """
    result = copy.deepcopy(master_data)

    personal_info = dict(result.get("personalInfo") or {})
    chosen_title = (role_title or content.get("title") or "").strip()
    if chosen_title:
        personal_info["title"] = chosen_title
    result["personalInfo"] = personal_info

    result["summary"] = scrub_em_dashes(str(content.get("summary", "")).strip())

    by_index = {
        item["index"]: item
        for item in content.get("workExperience", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    work_entries = result.get("workExperience") or []
    for index, entry in enumerate(work_entries):
        if not isinstance(entry, dict):
            continue
        generated = by_index.get(index)
        if generated is None:
            continue
        entry["title"] = scrub_em_dashes(str(generated.get("title", "")).strip())
        bullets = [
            scrub_em_dashes(str(b).strip())
            for b in generated.get("description", [])
            if str(b).strip()
        ]
        if max_bullets_per_entry is not None:
            bullets = bullets[:max_bullets_per_entry]
        entry["description"] = bullets
        entry["descriptionStyles"] = ["bullet"] * len(bullets)

    seen_skills: set[str] = set()
    skills: list[str] = []
    for raw in content.get("technicalSkills", []):
        skill = scrub_em_dashes(str(raw).strip())
        key = skill.casefold()
        if skill and key not in seen_skills:
            seen_skills.add(key)
            skills.append(skill)
    additional = dict(result.get("additional") or {})
    additional["technicalSkills"] = skills
    result["additional"] = additional

    return ResumeData.model_validate(result).model_dump()


# ---------------------------------------------------------------------------
# Style/grounding checks and the single repair pass
# ---------------------------------------------------------------------------


def _generated_text_corpus(content: dict[str, Any]) -> str:
    parts = [str(content.get("title", "")), str(content.get("summary", ""))]
    for item in content.get("workExperience", []):
        if isinstance(item, dict):
            parts.append(str(item.get("title", "")))
            parts.extend(str(b) for b in item.get("description", []))
    parts.extend(str(s) for s in content.get("technicalSkills", []))
    return "\n".join(parts)


def _already_grounded(term: str, grounding_text: str) -> bool:
    return term.strip().casefold() in grounding_text.casefold()


def _overlap_issue(
    sample_bullet: str, generated_text: str, language: str, grounding_text: str
) -> str | None:
    """A verbatim chunk of ``sample_bullet`` found in ``generated_text``, else None."""
    lowered_generated = generated_text.casefold()
    if language == "zh":
        chars = [c for c in sample_bullet if not c.isspace()]
        if len(chars) < _ZH_OVERLAP_CHARS:
            return None
        for start in range(len(chars) - _ZH_OVERLAP_CHARS + 1):
            chunk = "".join(chars[start : start + _ZH_OVERLAP_CHARS])
            if chunk and chunk.casefold() in lowered_generated:
                if not _already_grounded(chunk, grounding_text):
                    return chunk
        return None
    words = _EN_WORD_RE.findall(sample_bullet)
    if len(words) < _EN_OVERLAP_WORDS:
        return None
    for start in range(len(words) - _EN_OVERLAP_WORDS + 1):
        phrase = " ".join(words[start : start + _EN_OVERLAP_WORDS])
        if phrase.casefold() in lowered_generated:
            if not _already_grounded(phrase, grounding_text):
                return phrase
    return None


def check_created_style(
    content: dict[str, Any],
    profile: StyleProfile,
    targets: list[int],
    *,
    jd_text: str,
    master_text: str,
    same_language: bool,
) -> tuple[list[CreateCheckItem], list[CreateCheckItem]]:
    """Hard (repair-worthy) and soft (informational) style/grounding checks."""
    hard: list[CreateCheckItem] = []
    soft: list[CreateCheckItem] = []
    grounding_text = f"{jd_text}\n{master_text}"
    generated_text = _generated_text_corpus(content)

    by_index = {
        item["index"]: item
        for item in content.get("workExperience", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    all_bullets: list[str] = []
    for index, target in enumerate(targets):
        entry = by_index.get(index)
        bullets = entry.get("description", []) if entry else []
        bullets = [b for b in bullets if isinstance(b, str)]
        all_bullets.extend(bullets)
        if abs(len(bullets) - target) > 1:
            hard.append(
                CreateCheckItem(
                    code="bullet_count_mismatch",
                    severity="error",
                    path=f"workExperience[{index}].description",
                    detail=(
                        f"Entry {index} has {len(bullets)} bullets; the style "
                        f"sample suggests {target}."
                    ),
                )
            )

    skills = [s for s in content.get("technicalSkills", []) if isinstance(s, str)]
    if (
        profile.skills_grouped
        and len(skills) > 3
        and not any(_SKILL_LABEL_RE.match(s) for s in skills)
    ):
        hard.append(
            CreateCheckItem(
                code="skills_not_grouped",
                severity="error",
                path="technicalSkills",
                detail=(
                    "The style sample groups skills into labeled rows "
                    '("Label: a, b"); the generated skills list is flat.'
                ),
            )
        )

    for term in profile.protected_terms:
        if term.casefold() in generated_text.casefold() and not _already_grounded(
            term, grounding_text
        ):
            hard.append(
                CreateCheckItem(
                    code="sample_overlap",
                    severity="error",
                    path=None,
                    detail=f'Reused a name from the style sample: "{term}"',
                )
            )

    for number in profile.distinctive_numbers:
        if number in generated_text and not _already_grounded(number, grounding_text):
            hard.append(
                CreateCheckItem(
                    code="sample_overlap",
                    severity="error",
                    path=None,
                    detail=f'Reused a number from the style sample: "{number}"',
                )
            )

    for sample_bullet in profile.sample_bullets:
        overlap = _overlap_issue(
            sample_bullet, generated_text, profile.language, grounding_text
        )
        if overlap:
            hard.append(
                CreateCheckItem(
                    code="sample_overlap",
                    severity="error",
                    path=None,
                    detail=f'Copied text from the style sample: "{overlap}"',
                )
            )

    if all_bullets:
        quantified = sum(1 for b in all_bullets if re.search(r"\d", b))
        generated_ratio = quantified / len(all_bullets)
        if profile.quantified_ratio - generated_ratio > 0.25:
            soft.append(
                CreateCheckItem(
                    code="low_quantification",
                    severity="warning",
                    path=None,
                    detail=(
                        f"Only {generated_ratio:.0%} of bullets include a number; "
                        f"the style sample has {profile.quantified_ratio:.0%}."
                    ),
                )
            )
        if same_language:
            low = profile.bullet_length_min * 0.75
            high = profile.bullet_length_max * 1.25
            for index, entry in by_index.items():
                for b_index, bullet in enumerate(entry.get("description", [])):
                    if not isinstance(bullet, str):
                        continue
                    length = _bullet_length(bullet, profile.language)
                    if length < low or length > high:
                        soft.append(
                            CreateCheckItem(
                                code="length_drift",
                                severity="warning",
                                path=f"workExperience[{index}].description[{b_index}]",
                                detail=(
                                    f"Bullet is {length} units long; the style "
                                    f"sample's bullets are {profile.bullet_length_min} "
                                    f"to {profile.bullet_length_max}."
                                ),
                            )
                        )

    summary_sentences = len(
        [
            p
            for p in re.split(r"[。！？.!?]+", str(content.get("summary", "")))
            if p.strip()
        ]
    )
    if abs(summary_sentences - profile.summary_sentences) > 1:
        soft.append(
            CreateCheckItem(
                code="summary_length_drift",
                severity="warning",
                path="summary",
                detail=(
                    f"Summary has {summary_sentences} sentences; the style "
                    f"sample has {profile.summary_sentences}."
                ),
            )
        )

    return hard, soft


def _render_problems(hard: list[CreateCheckItem]) -> str:
    return "\n".join(f"- {item.detail}" for item in hard)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def generate_created_content(
    *,
    master_data: dict[str, Any],
    job_description: str,
    job_keywords: dict[str, Any],
    sample_text: str,
    profile: StyleProfile,
    language: str,
    max_bullets_per_entry: int | None,
) -> tuple[dict[str, Any], list[CreateCheckItem]]:
    """Generate create-mode content, check it, repair once if needed.

    Returns the LLM's content dict (not yet merged onto the master — call
    :func:`merge_created_content` next) and the style/grounding checks.
    """
    from app.services.refiner import _extract_all_text

    skeleton = build_work_skeleton(master_data)
    expected_indices = {entry["index"] for entry in skeleton}
    targets = bullet_targets(profile, len(skeleton), max_bullets_per_entry)
    same_language = profile.language == language

    sanitized_jd = _sanitize_user_input(job_description)
    output_language = get_language_name(language)
    style_rules = render_style_rules(profile, targets, same_language=same_language)
    skeleton_text = render_work_skeleton(skeleton, targets)
    keywords_str = _prepare_keywords_for_prompt(job_keywords)
    max_bullets_rule = (
        f"Write at most {max_bullets_per_entry} bullets per entry, close to "
        "the target bullet count listed above for each entry."
        if max_bullets_per_entry is not None
        else "Write close to the target bullet count listed above for each entry."
    )

    def validator(result: dict[str, Any]) -> dict[str, Any]:
        return _validate_created_shape(result, expected_indices)

    prompt = CREATE_RESUME_PROMPT.format(
        output_language=output_language,
        style_rules=style_rules or "- (no specific style rules)",
        style_sample=sample_text,
        job_description=sanitized_jd,
        job_keywords=keywords_str,
        work_skeleton=skeleton_text,
        max_bullets_rule=max_bullets_rule,
    )
    result = await complete_json(
        prompt=prompt,
        system_prompt="You are an expert resume writer. Output only valid JSON.",
        max_tokens=8192,
        response_validator=validator,
    )

    master_text = _extract_all_text(master_data)
    hard, soft = check_created_style(
        result,
        profile,
        targets,
        jd_text=sanitized_jd,
        master_text=master_text,
        same_language=same_language,
    )

    if not hard:
        return result, soft

    logger.info("Create mode: repairing %d hard style issue(s)", len(hard))
    import json as _json

    repair_prompt = CREATE_RESUME_REPAIR_PROMPT.format(
        output_language=output_language,
        problems=_render_problems(hard),
        style_rules=style_rules or "- (no specific style rules)",
        style_sample=sample_text,
        job_description=sanitized_jd,
        work_skeleton=skeleton_text,
        previous_draft=_json.dumps(result, ensure_ascii=False),
    )
    try:
        repaired = await complete_json(
            prompt=repair_prompt,
            system_prompt="You are an expert resume writer. Output only valid JSON.",
            max_tokens=8192,
            response_validator=validator,
        )
    except Exception as e:
        logger.warning("Create mode repair call failed, keeping first draft: %s", e)
        return result, hard + soft

    hard2, soft2 = check_created_style(
        repaired,
        profile,
        targets,
        jd_text=sanitized_jd,
        master_text=master_text,
        same_language=same_language,
    )
    checks = hard2 + soft2
    checks.append(
        CreateCheckItem(
            code="repair_applied",
            severity="info",
            path=None,
            detail="The draft was regenerated once to fix style or grounding issues.",
        )
    )
    return repaired, checks
