"""Layout measurement: fill classification, probe parsing and row-to-path mapping."""

import pytest

from app.schemas.models import PageFitSettings, TemplateSettings
from app.services.page_fit import (
    FILL_MIN,
    LayoutMeasurement,
    PdfLayout,
    RowLayout,
    classify_fill,
    description_row_paths,
    map_rows_to_paths,
    merge_layout,
    page_fit_from_template_settings,
    row_text_key,
)


@pytest.mark.parametrize(
    ("pages", "fill", "target", "expected"),
    [
        (1, 0.95, 1, "fit"),
        (1, FILL_MIN, 1, "fit"),  # exactly on the threshold counts as full
        (1, FILL_MIN - 0.01, 1, "under"),
        (1, 0.2, 2, "under"),  # fewer pages than the target
        (2, 0.1, 2, "under"),  # target page count but an empty last page
        (3, 0.5, 2, "over"),
        (2, 0.05, 1, "over"),  # a spill of one line is still over
        (5, 0.0, None, "fit"),  # no target (unlimited) always fits
    ],
)
def test_classify_fill(
    pages: int, fill: float, target: int | None, expected: str
) -> None:
    assert classify_fill(pages, fill, target) == expected


def _row(index: int, text: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "index": index,
        "style": "bullet",
        "text": text,
        "lines": 2,
        "cjk": False,
        "units": 20,
        "lineUnits": [12, 8],
        "unitsPerLine": 12,
        "capacityKnown": True,
    }
    base.update(overrides)
    return base


DATA = {
    "workExperience": [
        {"description": ["Cut <strong>latency</strong> by 40%", "Led the team"]},
    ],
    "personalProjects": [{"description": ["Shipped a CLI"]}],
    "customSections": {"pubs": {"items": [{"description": ["A paper"]}]}},
}


def test_description_rows_follow_the_page_order() -> None:
    assert [path for path, _ in description_row_paths(DATA)] == [
        "workExperience[0].description[0]",
        "workExperience[0].description[1]",
        "personalProjects[0].description[0]",
        "customSections.pubs.items[0].description[0]",
    ]


def test_row_text_key_ignores_markup_entities_and_spacing() -> None:
    assert row_text_key("Cut <strong>latency</strong>  by&nbsp;40%<br/>today") == (
        row_text_key("Cut latency by 40% today")
    )


def test_merge_layout_without_a_probe_keeps_only_the_pdf_numbers() -> None:
    layout = merge_layout(PdfLayout(pages=2, last_page_fill=0.4), None, DATA)
    assert (layout.pages, layout.last_page_fill) == (2, 0.4)
    assert layout.probe_ok is False
    assert layout.rows == ()


def test_merge_layout_maps_rows_and_skips_malformed_ones() -> None:
    probe = {
        "pageHeight": 1000,
        "contentWidth": 700,
        "contentHeight": 1500,
        "breakWaste": 12,
        "lineHeight": 18,
        "pageCount": 2,
        "rows": [
            _row(0, "Cut latency by 40%"),
            "not a row",
            {"index": 1},  # no text
            _row(2, "Led the team", lines=1, lineUnits=[3], capacityKnown=False),
            _row(3, "Text nobody has"),
        ],
    }
    layout = merge_layout(PdfLayout(pages=2, last_page_fill=0.4), probe, DATA)
    assert layout.probe_ok is True
    assert layout.dom_pages == 2
    assert layout.line_height_px == 18
    paths = [row.path for row in layout.rows]
    assert paths == [
        "workExperience[0].description[0]",
        "workExperience[0].description[1]",
        None,  # a row the data does not contain keeps no path
    ]
    assert set(layout.rows_by_path()) == {
        "workExperience[0].description[0]",
        "workExperience[0].description[1]",
    }


def test_equal_rows_take_their_paths_in_order() -> None:
    data = {"workExperience": [{"description": ["Same", "Same"]}]}
    rendered = [
        RowLayout(0, "bullet", "Same", 1, False, 1, (1,), 1.0, False),
        RowLayout(1, "bullet", "Same", 1, False, 1, (1,), 1.0, False),
    ]
    mapped = map_rows_to_paths(rendered, data)
    assert [row.path for row in mapped] == [
        "workExperience[0].description[0]",
        "workExperience[0].description[1]",
    ]


def test_capacity_is_the_median_of_wrapped_rows_of_that_script() -> None:
    def row(units_per_line: float, *, cjk: bool, known: bool) -> RowLayout:
        return RowLayout(0, "bullet", "x", 2, cjk, 10, (5, 5), units_per_line, known)

    layout = LayoutMeasurement(
        pages=1,
        last_page_fill=0.5,
        rows=(
            row(10, cjk=False, known=True),
            row(14, cjk=False, known=True),
            row(99, cjk=False, known=False),  # a one-line row is only a lower bound
            row(30, cjk=True, known=True),
        ),
    )
    assert layout.capacity(False) == 12
    assert layout.capacity(True) == 30
    assert LayoutMeasurement(pages=1, last_page_fill=0.5).capacity(False) is None


def test_saved_formatting_carries_the_description_format() -> None:
    saved = TemplateSettings.model_validate({"descriptionFormat": "text"})
    assert page_fit_from_template_settings(saved).descriptionFormat == "text"
    stored = {"descriptionFormat": "text", "seen": True}  # extra keys are tolerated
    assert page_fit_from_template_settings(stored).descriptionFormat == "text"
    assert page_fit_from_template_settings(None).descriptionFormat == "bullet"


def test_invalid_saved_formatting_falls_back_to_the_defaults() -> None:
    fit = page_fit_from_template_settings({"descriptionFormat": "nonsense"}, lang="en")
    assert fit == PageFitSettings(lang="en")


def test_description_format_reaches_the_print_query() -> None:
    assert PageFitSettings(descriptionFormat="text").to_query()[
        "descriptionFormat"
    ] == ("text")
