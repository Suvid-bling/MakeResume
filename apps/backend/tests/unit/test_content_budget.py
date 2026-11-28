"""Content budget: how much each work entry has to grow or shrink for a page target."""

from typing import Any

from app.schemas.models import ResumeChange
from app.services.budget_acceptance import (
    REASON_NOT_BUDGETED,
    REASON_ROW_COUNT,
    accept_rewrite_rows,
    rewrites_from_changes,
)
from app.services.content_budget import (
    plan_budget,
    render_budget_block,
)
from app.services.page_fit import LayoutMeasurement

PATH = "workExperience[0].description"


def _data(rows: int = 8) -> dict[str, Any]:
    return {
        "workExperience": [
            {
                "id": 1,
                "title": "Engineer",
                "company": "Acme",
                "years": "2020 - 2024",
                "description": [
                    f"Built service {i} that cut latency by {10 + i} percent for billing"
                    for i in range(rows)
                ],
                "descriptionStyles": ["bullet"] * rows,
            }
        ],
        "additional": {"technicalSkills": ["Python"]},
    }


def _layout(pages: int, fill: float) -> LayoutMeasurement:
    return LayoutMeasurement(pages=pages, last_page_fill=fill)


def test_a_resume_inside_the_tolerance_keeps_its_length() -> None:
    budget = plan_budget(_data(), _layout(1, 0.95), 1)
    assert budget.status == "fit"
    assert budget.delta_lines == 0
    entry = budget.entry(PATH)
    assert entry is not None
    assert (entry.mode, entry.target_rows) == ("keep", 8)
    assert budget.budgeted_paths() == set()


def test_no_page_target_keeps_every_entry() -> None:
    budget = plan_budget(_data(), _layout(3, 0.5), None)
    assert budget.status == "unlimited"
    assert budget.entry(PATH).mode == "keep"  # type: ignore[union-attr]


def test_an_under_filled_last_page_expands_within_the_row_limits() -> None:
    budget = plan_budget(_data(), _layout(1, 0.4), 1)
    entry = budget.entry(PATH)
    assert budget.status == "under" and budget.delta_lines > 0
    assert entry is not None and entry.mode == "expand"
    assert entry.current_rows < entry.target_rows <= entry.max_rows
    assert PATH in budget.budgeted_paths()
    assert not budget.drops


def test_over_the_target_drops_the_least_relevant_rows_first() -> None:
    data = _data()
    budget = plan_budget(data, _layout(2, 0.3), 1)
    assert budget.status == "over" and budget.delta_lines < 0
    # Cutting more than 40% of the rows drops whole bullets before any rewrite.
    assert budget.drops
    trimmed = budget.apply_drops(data)
    kept = trimmed["workExperience"][0]["description"]
    assert len(kept) == 8 - len(budget.drops)
    assert len(kept) >= 1  # never below one row per entry
    assert budget.entry(PATH).current_rows == len(kept)  # type: ignore[union-attr]


def test_lines_per_page_depends_on_the_page_not_on_the_resume_length() -> None:
    def probed(pages: int, content_height: float) -> LayoutMeasurement:
        return LayoutMeasurement(
            pages=pages,
            last_page_fill=0.5,
            page_height_px=1000.0,
            content_height_px=content_height,  # the whole resume: grows with pages
            line_height_px=20.0,
            probe_ok=True,
        )

    one_page = plan_budget(_data(), probed(1, 600.0), 1)
    three_pages = plan_budget(_data(), probed(3, 2500.0), 3)
    assert one_page.reliable and three_pages.reliable
    assert one_page.lines_per_page == three_pages.lines_per_page == 50.0


def test_a_layout_without_a_page_height_falls_back_to_the_default() -> None:
    layout = LayoutMeasurement(
        pages=1,
        last_page_fill=0.5,
        content_height_px=900.0,
        line_height_px=20.0,
        probe_ok=True,  # answered, but reported no page height
    )
    budget = plan_budget(_data(), layout, 1)
    assert budget.reliable is False
    assert budget.lines_per_page == 50.0


def test_the_budget_block_names_the_entry_and_its_row_target() -> None:
    budget = plan_budget(_data(), _layout(1, 0.4), 1)
    block = render_budget_block(budget)
    entry = budget.entry(PATH)
    assert entry is not None
    assert entry.label in block
    assert str(entry.target_rows) in block


def _expand_budget() -> tuple[Any, dict[str, Any]]:
    data = _data()
    budget = plan_budget(data, _layout(1, 0.4), 1)
    return budget, data


def test_rewrite_with_the_planned_row_count_is_merged_and_marked_up() -> None:
    budget, data = _expand_budget()
    entry = budget.entry(PATH)
    assert entry is not None
    rows = [f"Built service {i} cutting latency by {10 + i} percent" for i in range(8)]
    rows += [f"-> Latency fell {10 + i} percent" for i in range(entry.target_rows - 8)]
    result = accept_rewrite_rows(data, {PATH: rows}, budget, master=data)
    assert result.accepted == [PATH] and not result.reverted
    merged = result.data["workExperience"][0]
    assert len(merged["description"]) == entry.target_rows
    # An arrow prefix is stripped from the text and becomes a result row.
    assert not any(row.startswith("->") for row in merged["description"])
    assert merged["descriptionStyles"].count("result") == entry.target_rows - 8


def test_rewrite_far_from_the_planned_row_count_is_reverted() -> None:
    budget, data = _expand_budget()
    result = accept_rewrite_rows(data, {PATH: ["Only one row"]}, budget, master=data)
    assert result.reverted == {PATH: REASON_ROW_COUNT}
    assert result.data == data


def test_rewrite_of_an_entry_that_is_not_budgeted_is_reverted() -> None:
    data = _data()
    budget = plan_budget(data, _layout(1, 0.95), 1)  # fits, so every entry is kept
    result = accept_rewrite_rows(data, {PATH: ["x"] * 8}, budget, master=data)
    assert result.reverted == {PATH: REASON_NOT_BUDGETED}


def test_rewrite_that_invents_a_number_is_reverted_in_tailor_mode_only() -> None:
    budget, data = _expand_budget()
    entry = budget.entry(PATH)
    assert entry is not None
    rows = [f"Built service {i} cutting latency by 777 percent" for i in range(8)]
    rows += ["Kept the platform steady"] * (entry.target_rows - 8)
    tailor = accept_rewrite_rows(data, {PATH: rows}, budget, master=data)
    assert PATH in tailor.reverted
    created = accept_rewrite_rows(
        data, {PATH: rows}, budget, master=data, create_mode=True
    )
    assert created.accepted == [PATH]


def test_rewrites_are_read_from_diff_changes() -> None:
    change = ResumeChange(
        path=PATH, action="rewrite_rows", original=None, value=["a", "b"], reason="fit"
    )
    plain = {"action": "replace", "path": "summary", "value": "x"}
    assert rewrites_from_changes([change, plain]) == {PATH: ["a", "b"]}
