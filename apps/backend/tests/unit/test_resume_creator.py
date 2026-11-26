"""Shape validation, merge and style/grounding checks for "create" mode."""

from __future__ import annotations

from typing import Any

import pytest

from app.prompts.style import get_style_sample
from app.services.resume_creator import (
    _validate_created_shape,
    build_work_skeleton,
    check_created_style,
    merge_created_content,
    render_style_rules,
)
from app.services.style_profile import StyleProfile, compute_style_profile


def _master() -> dict[str, Any]:
    return {
        "personalInfo": {
            "name": "Ada Lovelace",
            "title": "Software Engineer",
            "email": "ada@example.com",
            "phone": "555-0100",
            "location": "Remote",
        },
        "summary": "Original summary about Ada's work.",
        "workExperience": [
            {
                "id": 1,
                "title": "Software Engineer",
                "company": "Alpha Corp",
                "location": "Remote",
                "years": "Jan 2022 - Present",
                "description": ["Did a thing", "Did another thing"],
                "descriptionStyles": ["bullet", "bullet"],
            },
            {
                "id": 2,
                "title": "Junior Engineer",
                "company": "Beta Inc",
                "location": "NYC",
                "years": "Jan 2020 - Dec 2021",
                "description": ["Wrote code"],
                "descriptionStyles": ["bullet"],
            },
        ],
        "education": [
            {
                "id": 1,
                "institution": "Example University",
                "degree": "BSc",
                "years": "2016 - 2020",
                "description": "Studied things",
            }
        ],
        "personalProjects": [
            {
                "id": 1,
                "name": "Side Project",
                "role": "Maintainer",
                "years": "2021",
                "description": ["Built a thing"],
                "descriptionStyles": ["bullet"],
            }
        ],
        "additional": {
            "technicalSkills": ["Python", "SQL"],
            "languages": ["English"],
            "certificationsTraining": ["Some Cert"],
            "awards": ["Some Award"],
        },
        "customSections": {},
        "sectionMeta": [],
    }


def _content(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Backend Engineer",
        "summary": "New summary written for the job.",
        "workExperience": [
            {
                "index": 0,
                "title": "Senior Engineer",
                "description": ["Shipped X", "Shipped Y"],
            },
            {"index": 1, "title": "Engineer", "description": ["Built Z"]},
        ],
        "technicalSkills": ["Go", "Kubernetes"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_work_skeleton
# ---------------------------------------------------------------------------


def test_build_work_skeleton_lists_entries_in_order() -> None:
    skeleton = build_work_skeleton(_master())
    assert [e["index"] for e in skeleton] == [0, 1]
    assert skeleton[0]["company"] == "Alpha Corp"
    assert skeleton[0]["location"] == "Remote"
    assert skeleton[0]["years"] == "Jan 2022 - Present"
    assert skeleton[1]["company"] == "Beta Inc"


def test_build_work_skeleton_empty_for_no_experience() -> None:
    master = _master()
    master["workExperience"] = []
    assert build_work_skeleton(master) == []


# ---------------------------------------------------------------------------
# _validate_created_shape
# ---------------------------------------------------------------------------


def test_validate_created_shape_accepts_well_formed_result() -> None:
    result = _validate_created_shape(_content(), {0, 1})
    assert result["title"] == "Backend Engineer"


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"summary": "   "},
        {"technicalSkills": []},
        {"technicalSkills": ["", "  "]},
    ],
)
def test_validate_created_shape_rejects_blank_required_fields(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _validate_created_shape(_content(**overrides), {0, 1})


def test_validate_created_shape_rejects_missing_index() -> None:
    content = _content(
        workExperience=[{"index": 0, "title": "Engineer", "description": ["A"]}]
    )
    with pytest.raises(ValueError):
        _validate_created_shape(content, {0, 1})


def test_validate_created_shape_rejects_duplicate_index() -> None:
    content = _content(
        workExperience=[
            {"index": 0, "title": "Engineer", "description": ["A"]},
            {"index": 0, "title": "Engineer 2", "description": ["B"]},
        ]
    )
    with pytest.raises(ValueError):
        _validate_created_shape(content, {0, 1})


def test_validate_created_shape_rejects_empty_description() -> None:
    content = _content(
        workExperience=[
            {"index": 0, "title": "Engineer", "description": []},
            {"index": 1, "title": "Engineer", "description": ["A"]},
        ]
    )
    with pytest.raises(ValueError):
        _validate_created_shape(content, {0, 1})


def test_validate_created_shape_allows_empty_work_when_master_has_none() -> None:
    content = _content(workExperience=[])
    result = _validate_created_shape(content, set())
    assert result["workExperience"] == []


# ---------------------------------------------------------------------------
# merge_created_content
# ---------------------------------------------------------------------------


def test_merge_preserves_company_location_years_entry_count() -> None:
    master = _master()
    merged = merge_created_content(
        master, _content(), role_title="Backend Engineer", max_bullets_per_entry=None
    )
    assert len(merged["workExperience"]) == 2
    assert merged["workExperience"][0]["company"] == "Alpha Corp"
    assert merged["workExperience"][0]["location"] == "Remote"
    assert merged["workExperience"][0]["years"] == "Jan 2022 - Present"
    assert merged["workExperience"][1]["company"] == "Beta Inc"


def test_merge_replaces_title_summary_bullets_and_skills() -> None:
    master = _master()
    merged = merge_created_content(
        master, _content(), role_title="Backend Engineer", max_bullets_per_entry=None
    )
    assert merged["personalInfo"]["title"] == "Backend Engineer"
    assert merged["summary"] == "New summary written for the job."
    assert merged["workExperience"][0]["title"] == "Senior Engineer"
    assert merged["workExperience"][0]["description"] == ["Shipped X", "Shipped Y"]
    assert merged["workExperience"][0]["descriptionStyles"] == ["bullet", "bullet"]
    assert merged["additional"]["technicalSkills"] == ["Go", "Kubernetes"]


def test_merge_falls_back_to_llm_title_when_role_title_empty() -> None:
    master = _master()
    merged = merge_created_content(
        master, _content(title="LLM Title"), role_title="", max_bullets_per_entry=None
    )
    assert merged["personalInfo"]["title"] == "LLM Title"


def test_merge_preserves_personal_info_other_fields() -> None:
    master = _master()
    merged = merge_created_content(
        master, _content(), role_title="New Title", max_bullets_per_entry=None
    )
    assert merged["personalInfo"]["name"] == "Ada Lovelace"
    assert merged["personalInfo"]["email"] == "ada@example.com"


def test_merge_preserves_education_projects_additional_lists() -> None:
    master = _master()
    merged = merge_created_content(
        master, _content(), role_title="Backend Engineer", max_bullets_per_entry=None
    )
    assert merged["education"] == master["education"]
    assert merged["personalProjects"][0]["name"] == "Side Project"
    assert merged["personalProjects"][0]["description"] == ["Built a thing"]
    assert merged["additional"]["languages"] == ["English"]
    assert merged["additional"]["certificationsTraining"] == ["Some Cert"]
    assert merged["additional"]["awards"] == ["Some Award"]


def test_merge_truncates_bullets_to_cap() -> None:
    master = _master()
    content = _content(
        workExperience=[
            {
                "index": 0,
                "title": "Senior Engineer",
                "description": ["A", "B", "C", "D"],
            },
            {"index": 1, "title": "Engineer", "description": ["A"]},
        ]
    )
    merged = merge_created_content(
        master, content, role_title="X", max_bullets_per_entry=2
    )
    assert merged["workExperience"][0]["description"] == ["A", "B"]
    assert len(merged["workExperience"][0]["descriptionStyles"]) == 2


def test_merge_scrubs_em_dash() -> None:
    master = _master()
    content = _content(summary="Led growth—a lot of it.")
    merged = merge_created_content(
        master, content, role_title="X", max_bullets_per_entry=None
    )
    assert "—" not in merged["summary"]


def test_merge_dedupes_skills_case_insensitively() -> None:
    master = _master()
    content = _content(technicalSkills=["Python", "python", "SQL", " SQL "])
    merged = merge_created_content(
        master, content, role_title="X", max_bullets_per_entry=None
    )
    assert merged["additional"]["technicalSkills"] == ["Python", "SQL"]


def test_merge_handles_master_with_no_work_experience() -> None:
    master = _master()
    master["workExperience"] = []
    content = _content(workExperience=[])
    merged = merge_created_content(
        master, content, role_title="X", max_bullets_per_entry=None
    )
    assert merged["workExperience"] == []
    assert merged["summary"] == "New summary written for the job."


# ---------------------------------------------------------------------------
# render_style_rules
# ---------------------------------------------------------------------------


def test_render_style_rules_drops_length_and_structure_rules_cross_language() -> None:
    sample = get_style_sample("senior_fullstack_zh")
    assert sample is not None
    profile = compute_style_profile(sample)
    same_lang = render_style_rules(profile, [4, 3, 1], same_language=True)
    cross_lang = render_style_rules(profile, [4, 3, 1], same_language=False)
    assert "characters" in same_lang
    assert "characters" not in cross_lang
    assert "sentence structure" not in cross_lang
    # Length-independent rules survive the language switch.
    assert "bullet count" in same_lang and "bullet count" in cross_lang


# ---------------------------------------------------------------------------
# check_created_style
# ---------------------------------------------------------------------------


def _synthetic_profile(**overrides: Any) -> StyleProfile:
    base = dict(
        sample_id="synthetic",
        language="en",
        summary_sentences=2,
        bullets_per_entry=(3, 2),
        quantified_ratio=1.0,
        bullet_length_min=10,
        bullet_length_max=20,
        skills_grouped=False,
        skills_separator=", ",
        protected_terms=frozenset({"Acme Corp"}),
        distinctive_numbers=frozenset({"2,400"}),
        sample_bullets=(
            "Led the migration of the legacy billing platform to microservices",
        ),
    )
    base.update(overrides)
    return StyleProfile(**base)


def test_check_created_style_flags_bullet_count_mismatch() -> None:
    profile = _synthetic_profile()
    content = _content(
        workExperience=[
            {"index": 0, "title": "A", "description": ["only one bullet here"]},
            {"index": 1, "title": "B", "description": ["x", "y"]},
        ]
    )
    hard, _soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    codes = [c.code for c in hard]
    assert "bullet_count_mismatch" in codes
    assert any(c.path == "workExperience[0].description" for c in hard)


def test_check_created_style_flags_unflagged_skills_not_grouped() -> None:
    profile = _synthetic_profile(skills_grouped=True, skills_separator=", ")
    content = _content(technicalSkills=["Go", "Kubernetes", "Docker", "Python"])
    hard, _soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "skills_not_grouped" for c in hard)


def test_check_created_style_accepts_grouped_skills() -> None:
    profile = _synthetic_profile(skills_grouped=True, skills_separator=", ")
    content = _content(technicalSkills=["Languages: Go, Python", "Infra: Docker, K8s"])
    hard, _soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    assert not any(c.code == "skills_not_grouped" for c in hard)


def test_check_created_style_flags_reused_protected_term() -> None:
    profile = _synthetic_profile()
    content = _content(summary="Worked closely with Acme Corp on a major project.")
    hard, _soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "sample_overlap" and "Acme Corp" in c.detail for c in hard)


def test_check_created_style_allows_protected_term_when_grounded_in_jd() -> None:
    profile = _synthetic_profile()
    content = _content(summary="Worked closely with Acme Corp on a major project.")
    hard, _soft = check_created_style(
        content,
        profile,
        targets=[3, 2],
        jd_text="This role is at Acme Corp.",
        master_text="",
        same_language=True,
    )
    assert not any(c.code == "sample_overlap" and "Acme Corp" in c.detail for c in hard)


def test_check_created_style_flags_reused_distinctive_number() -> None:
    profile = _synthetic_profile()
    content = _content(summary="Supported 2,400 merchant accounts across regions.")
    hard, _soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "sample_overlap" and "2,400" in c.detail for c in hard)


def test_check_created_style_flags_verbatim_overlap_english() -> None:
    profile = _synthetic_profile()
    content = _content(
        workExperience=[
            {
                "index": 0,
                "title": "A",
                "description": [
                    "Led the migration of the legacy billing platform to microservices for a new client"
                ],
            },
            {"index": 1, "title": "B", "description": ["x", "y"]},
        ]
    )
    hard, _soft = check_created_style(
        content, profile, targets=[1, 2], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "sample_overlap" for c in hard)


def test_check_created_style_low_quantification_is_soft() -> None:
    profile = _synthetic_profile(quantified_ratio=1.0)
    content = _content(
        workExperience=[
            {"index": 0, "title": "A", "description": ["did a thing", "did another"]},
            {"index": 1, "title": "B", "description": ["did more"]},
        ]
    )
    hard, soft = check_created_style(
        content, profile, targets=[2, 1], jd_text="", master_text="", same_language=True
    )
    assert not any(c.code == "low_quantification" for c in hard)
    assert any(c.code == "low_quantification" for c in soft)


def test_check_created_style_length_drift_only_when_same_language() -> None:
    profile = _synthetic_profile(bullet_length_min=10, bullet_length_max=12)
    long_bullet = "word " * 40
    content = _content(
        workExperience=[
            {"index": 0, "title": "A", "description": [long_bullet.strip()]},
            {"index": 1, "title": "B", "description": ["short bullet here now"]},
        ]
    )
    hard_same, soft_same = check_created_style(
        content, profile, targets=[1, 1], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "length_drift" for c in soft_same)

    hard_cross, soft_cross = check_created_style(
        content,
        profile,
        targets=[1, 1],
        jd_text="",
        master_text="",
        same_language=False,
    )
    assert not any(c.code == "length_drift" for c in soft_cross)


def test_check_created_style_summary_length_drift() -> None:
    profile = _synthetic_profile(summary_sentences=1)
    content = _content(summary="One. Two. Three. Four sentences in a row.")
    hard, soft = check_created_style(
        content, profile, targets=[3, 2], jd_text="", master_text="", same_language=True
    )
    assert any(c.code == "summary_length_drift" for c in soft)
