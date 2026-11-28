"""Result rows ("-> ...") in the resume schema."""

import pytest

from app.schemas import ResumeData
from app.schemas.models import PageFitSettings, TemplateSettings


def _entry(description: list[str], styles: list[str] | None = None) -> dict:
    entry: dict = {
        "id": 1,
        "title": "Engineer",
        "company": "Acme",
        "description": description,
    }
    if styles is not None:
        entry["descriptionStyles"] = styles
    return {"workExperience": [entry]}


@pytest.mark.parametrize("arrow", ["->", "→", "➜"])
def test_an_arrow_prefix_becomes_a_result_row(arrow: str) -> None:
    data = ResumeData.model_validate(
        _entry(["Cut latency", f"{arrow} Latency fell 40%"])
    ).model_dump()
    job = data["workExperience"][0]
    assert job["description"] == ["Cut latency", "Latency fell 40%"]
    assert job["descriptionStyles"] == ["bullet", "result"]


def test_a_dash_prefix_stays_an_action_row() -> None:
    job = ResumeData.model_validate(_entry(["- Cut latency"])).model_dump()[
        "workExperience"
    ][0]
    assert job["description"] == ["Cut latency"]
    assert job["descriptionStyles"] == ["bullet"]


def test_an_explicit_plain_style_wins_over_an_arrow() -> None:
    job = ResumeData.model_validate(
        _entry(["-> Heading line"], ["plain"])
    ).model_dump()["workExperience"][0]
    assert job["descriptionStyles"] == ["plain"]


def test_stored_result_styles_are_kept_and_unknown_ones_become_bullets() -> None:
    job = ResumeData.model_validate(
        _entry(["a", "b", "c"], ["result", "weird", "plain"])
    ).model_dump()["workExperience"][0]
    assert job["descriptionStyles"] == ["result", "bullet", "plain"]


def test_description_format_defaults_to_bullets_and_rejects_other_values() -> None:
    assert TemplateSettings().descriptionFormat == "bullet"
    assert PageFitSettings().descriptionFormat == "bullet"
    assert TemplateSettings(descriptionFormat="text").descriptionFormat == "text"
    with pytest.raises(ValueError):
        PageFitSettings(descriptionFormat="markdown")  # type: ignore[arg-type]
