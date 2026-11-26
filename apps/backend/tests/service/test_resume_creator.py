"""Service tests for resume_creator.generate_created_content — LLM mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.prompts.style import get_style_sample
from app.services.resume_creator import generate_created_content
from app.services.style_profile import compute_style_profile


def _master() -> dict[str, Any]:
    return {
        "personalInfo": {"name": "Ada Lovelace", "title": "Engineer"},
        "summary": "Old summary.",
        "workExperience": [
            {
                "id": 1,
                "title": "Engineer",
                "company": "Alpha Corp",
                "location": "Remote",
                "years": "Jan 2022 - Present",
                "description": ["Old bullet"],
                "descriptionStyles": ["bullet"],
            }
        ],
        "education": [],
        "personalProjects": [],
        "additional": {
            "technicalSkills": ["Python"],
            "languages": [],
            "certificationsTraining": [],
            "awards": [],
        },
        "customSections": {},
        "sectionMeta": [],
    }


def _clean_draft() -> dict[str, Any]:
    return {
        "title": "Backend Engineer",
        "summary": "A clean two sentence summary. It mentions no sample facts.",
        "workExperience": [
            {
                "index": 0,
                "title": "Backend Engineer",
                "description": [
                    "Built an API",
                    "Shipped a feature",
                    "Reviewed code",
                    "Mentored a peer",
                ],
            }
        ],
        "technicalSkills": ["Go", "Kubernetes"],
    }


@pytest.fixture
def sample_and_profile():
    sample = get_style_sample("senior_fullstack_zh")
    assert sample is not None
    profile = compute_style_profile(sample)
    return sample, profile


@patch("app.services.resume_creator.complete_json", new_callable=AsyncMock)
async def test_generate_created_content_single_call_when_clean(
    mock_llm, sample_and_profile
):
    sample, profile = sample_and_profile
    mock_llm.return_value = _clean_draft()

    result, checks = await generate_created_content(
        master_data=_master(),
        job_description="We need a backend engineer with Go and Kubernetes experience.",
        job_keywords={"required_skills": ["Go", "Kubernetes"]},
        sample_text=sample.text,
        profile=profile,
        language="zh",
        max_bullets_per_entry=None,
    )

    assert mock_llm.call_count == 1
    assert result["title"] == "Backend Engineer"
    assert not any(c.severity == "error" for c in checks)


@patch("app.services.resume_creator.complete_json", new_callable=AsyncMock)
async def test_generate_created_content_prompt_contains_required_pieces(
    mock_llm, sample_and_profile
):
    sample, profile = sample_and_profile
    mock_llm.return_value = _clean_draft()

    await generate_created_content(
        master_data=_master(),
        job_description="JD mentions {curly} braces and Go.",
        job_keywords={"required_skills": ["Go"]},
        sample_text=sample.text,
        profile=profile,
        language="zh",
        max_bullets_per_entry=3,
    )

    prompt = mock_llm.call_args.kwargs["prompt"]
    assert sample.text.strip() in prompt
    assert "Alpha Corp" in prompt  # work skeleton
    assert "JD mentions" in prompt
    assert "at most 3 bullets" in prompt


@patch("app.services.resume_creator.complete_json", new_callable=AsyncMock)
async def test_generate_created_content_repairs_once_on_sample_overlap(
    mock_llm, sample_and_profile
):
    sample, profile = sample_and_profile
    dirty_draft = _clean_draft()
    dirty_draft["summary"] = "Worked at Brightmoor Commerce on a major project."
    mock_llm.side_effect = [dirty_draft, _clean_draft()]

    result, checks = await generate_created_content(
        master_data=_master(),
        job_description="We need a backend engineer.",
        job_keywords={},
        sample_text=sample.text,
        profile=profile,
        language="zh",
        max_bullets_per_entry=None,
    )

    assert mock_llm.call_count == 2
    repair_prompt = mock_llm.call_args_list[1].kwargs["prompt"]
    assert "Brightmoor Commerce" in repair_prompt
    assert result["summary"] == _clean_draft()["summary"]
    assert any(c.code == "repair_applied" for c in checks)
    assert not any(c.severity == "error" for c in checks)


@patch("app.services.resume_creator.complete_json", new_callable=AsyncMock)
async def test_generate_created_content_keeps_first_draft_if_repair_call_fails(
    mock_llm, sample_and_profile
):
    sample, profile = sample_and_profile
    dirty_draft = _clean_draft()
    dirty_draft["summary"] = "Worked at Brightmoor Commerce on a major project."
    mock_llm.side_effect = [dirty_draft, RuntimeError("LLM unavailable")]

    result, checks = await generate_created_content(
        master_data=_master(),
        job_description="We need a backend engineer.",
        job_keywords={},
        sample_text=sample.text,
        profile=profile,
        language="zh",
        max_bullets_per_entry=None,
    )

    assert result == dirty_draft
    assert any(c.code == "sample_overlap" for c in checks)
