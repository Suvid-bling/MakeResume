"""The "create" tailor mode: preview, confirm, and entry-point guards.

Real routers and a real temp database; every LLM boundary is mocked. This
mirrors the pattern in test_writing_loop_api.py, but mocks
``generate_created_content`` directly (its own behavior is covered by
tests/unit and tests/service/test_resume_creator.py) so these tests focus on
how the router wires the create flow: style-sample validation, the merge
onto the master skeleton, the writing loop, and the preview/confirm handoff.
"""

import copy
from typing import Any
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app

STYLE_SAMPLE_ID = "senior_fullstack_zh"


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _master_and_job(sample_resume: dict[str, Any]) -> tuple[str, str]:
    data = copy.deepcopy(sample_resume)
    with (
        patch(
            "app.routers.resumes.parse_document",
            new_callable=AsyncMock,
            return_value="# Jane Doe\n",
        ),
        patch(
            "app.routers.resumes.parse_resume_to_json",
            new_callable=AsyncMock,
            return_value=data,
        ),
    ):
        async with _client() as client:
            upload = await client.post(
                "/api/v1/resumes/upload",
                files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
    assert upload.status_code == 200, upload.text
    async with _client() as client:
        job = await client.post(
            "/api/v1/jobs/upload",
            json={
                "job_descriptions": [
                    "Senior backend engineer at Nimbus. Go, Kubernetes, distributed "
                    "systems. Own platform reliability and mentor the team."
                ]
            },
        )
    assert job.status_code == 200, job.text
    return upload.json()["resume_id"], job.json()["job_id"][0]


def _created_draft() -> dict[str, Any]:
    return {
        "title": "Senior Backend Engineer",
        "summary": "A platform engineer ready for distributed systems work. Strong Go background.",
        "workExperience": [
            {
                "index": 0,
                "title": "Senior Platform Engineer",
                "description": [
                    "Owned platform reliability for a Go-based distributed system",
                    "Mentored three engineers on Kubernetes operations",
                ],
            },
            {
                "index": 1,
                "title": "Platform Engineer",
                "description": [
                    "Operated Kubernetes clusters supporting production traffic"
                ],
            },
        ],
        "technicalSkills": ["Go", "Kubernetes", "Distributed Systems"],
    }


async def test_create_preview_requires_known_style_sample_id(
    isolated_db: Any, sample_resume: dict[str, Any]
) -> None:
    resume_id, job_id = await _master_and_job(sample_resume)
    async with _client() as client:
        resp = await client.post(
            "/api/v1/resumes/improve/preview",
            json={"resume_id": resume_id, "job_id": job_id, "prompt_id": "create"},
        )
    assert resp.status_code == 400

    async with _client() as client:
        resp = await client.post(
            "/api/v1/resumes/improve/preview",
            json={
                "resume_id": resume_id,
                "job_id": job_id,
                "prompt_id": "create",
                "style_sample_id": "not-a-real-sample",
            },
        )
    assert resp.status_code == 400


async def test_create_preview_requires_structured_master_data(isolated_db: Any) -> None:
    from app.database import db

    resume = await db.create_resume(
        content="# Jane Doe\nSome markdown resume with no structured data.",
        content_type="md",
        processing_status="ready",
        is_master=True,
    )
    async with _client() as client:
        job = await client.post(
            "/api/v1/jobs/upload",
            json={"job_descriptions": ["Backend engineer role."]},
        )
    job_id = job.json()["job_id"][0]

    async with _client() as client:
        resp = await client.post(
            "/api/v1/resumes/improve/preview",
            json={
                "resume_id": resume["resume_id"],
                "job_id": job_id,
                "prompt_id": "create",
                "style_sample_id": STYLE_SAMPLE_ID,
            },
        )
    assert resp.status_code == 400


async def test_create_mode_preview_then_confirm_end_to_end(
    isolated_db: Any, sample_resume: dict[str, Any]
) -> None:
    resume_id, job_id = await _master_and_job(sample_resume)

    with (
        patch(
            "app.routers.resumes.extract_job_keywords",
            new_callable=AsyncMock,
            return_value={"role": "Senior Backend Engineer", "required_skills": ["Go"]},
        ),
        patch(
            "app.routers.resumes.generate_created_content",
            new_callable=AsyncMock,
            return_value=(_created_draft(), []),
        ),
        patch(
            "app.services.writing_loop.complete_json",
            new_callable=AsyncMock,
            return_value={"bullets": []},
        ),
    ):
        async with _client() as client:
            resp = await client.post(
                "/api/v1/resumes/improve/preview",
                json={
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "prompt_id": "create",
                    "style_sample_id": STYLE_SAMPLE_ID,
                },
            )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    preview = data["resume_preview"]

    # Generated fields reflect the mocked draft.
    assert preview["personalInfo"]["title"] == "Senior Backend Engineer"
    assert preview["summary"].startswith("A platform engineer")
    assert preview["workExperience"][0]["title"] == "Senior Platform Engineer"
    assert preview["workExperience"][1]["title"] == "Platform Engineer"
    assert preview["additional"]["technicalSkills"] == [
        "Go",
        "Kubernetes",
        "Distributed Systems",
    ]

    # Master fields the create mode must not touch stay exactly as uploaded.
    assert preview["personalInfo"]["name"] == sample_resume["personalInfo"]["name"]
    assert preview["personalInfo"]["email"] == sample_resume["personalInfo"]["email"]
    assert len(preview["workExperience"]) == len(sample_resume["workExperience"])
    for original, result in zip(
        sample_resume["workExperience"], preview["workExperience"]
    ):
        assert result["company"] == original["company"]
        assert result["location"] == original["location"]
        assert result["years"] == original["years"]
    assert (
        preview["education"][0]["institution"]
        == sample_resume["education"][0]["institution"]
    )
    assert (
        preview["personalProjects"][0]["name"]
        == sample_resume["personalProjects"][0]["name"]
    )
    assert (
        preview["additional"]["languages"] == sample_resume["additional"]["languages"]
    )

    # The style sample is zh; this test's content language defaults to en, so
    # a language_mismatch info check is expected (and correct).
    assert [c["code"] for c in data["create_checks"]] == ["language_mismatch"]
    assert any("CREATE_MODE" in w for w in data["warnings"])

    with patch(
        "app.routers.resumes.generate_resume_title",
        new_callable=AsyncMock,
        return_value="Senior Backend Engineer - Nimbus",
    ):
        async with _client() as client:
            confirm = await client.post(
                "/api/v1/resumes/improve/confirm",
                json={
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "preview_id": data["preview_id"],
                    "improved_data": preview,
                    "improvements": data["improvements"],
                },
            )
    assert confirm.status_code == 200, confirm.text
    confirmed = confirm.json()["data"]
    assert confirmed["resume_id"]
    assert confirmed["resume_id"] != resume_id

    saved = await isolated_db.get_resume(confirmed["resume_id"])
    assert saved["is_master"] is False
    assert saved["parent_id"] == resume_id
    assert saved["processed_data"]["personalInfo"]["title"] == "Senior Backend Engineer"
    assert (
        saved["processed_data"]["workExperience"][0]["company"]
        == sample_resume["workExperience"][0]["company"]
    )

    # The master itself was never modified.
    master_after = await isolated_db.get_resume(resume_id)
    assert (
        master_after["processed_data"]["personalInfo"]["title"]
        == (sample_resume["personalInfo"]["title"])
    )


async def test_improve_endpoint_rejects_create_prompt_id(
    isolated_db: Any, sample_resume: dict[str, Any]
) -> None:
    resume_id, job_id = await _master_and_job(sample_resume)
    async with _client() as client:
        resp = await client.post(
            "/api/v1/resumes/improve",
            json={"resume_id": resume_id, "job_id": job_id, "prompt_id": "create"},
        )
    assert resp.status_code == 400


async def test_queue_endpoint_rejects_create_prompt_id(
    isolated_db: Any, sample_resume: dict[str, Any]
) -> None:
    resume_id, _job_id = await _master_and_job(sample_resume)
    async with _client() as client:
        resp = await client.post(
            "/api/v1/jobs/queue",
            json={
                "resume_id": resume_id,
                "job_description": "Some job description that is long enough.",
                "prompt_id": "create",
            },
        )
    assert resp.status_code == 400
