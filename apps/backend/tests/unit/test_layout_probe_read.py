"""Reading the print page's layout probe must never cost the PDF."""

import asyncio
from typing import Any

import pytest

from app import pdf


class _Page:
    """The two Playwright calls the probe reader makes."""

    def __init__(self, layout: Any, *, answers: bool = True) -> None:
        self._layout = layout
        self._answers = answers

    async def wait_for_function(self, *_args: Any, **_kwargs: Any) -> None:
        if not self._answers:
            await asyncio.sleep(60)

    async def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._layout


def _deadline(seconds: float = 30.0) -> float:
    return asyncio.get_running_loop().time() + seconds


async def test_a_published_layout_is_returned() -> None:
    layout = {"pageCount": 1, "rows": []}
    assert await pdf._read_layout_probe(_Page(layout), _deadline()) == layout


async def test_a_failure_report_gives_no_layout() -> None:
    page = _Page({"error": "probe-failed"})
    assert await pdf._read_layout_probe(page, _deadline()) is None


async def test_a_value_that_is_not_a_layout_gives_none() -> None:
    assert await pdf._read_layout_probe(_Page("oops"), _deadline()) is None


async def test_a_silent_page_costs_only_the_probe_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf, "_LAYOUT_PROBE_TIMEOUT_SECONDS", 0.2)
    loop = asyncio.get_running_loop()
    work_deadline = _deadline(30.0)
    started = loop.time()
    layout = await pdf._read_layout_probe(_Page(None, answers=False), work_deadline)
    assert layout is None
    waited = loop.time() - started
    assert waited < 2.0  # not the 30 s work deadline
    assert work_deadline - loop.time() > 25.0  # the PDF step still has its time


async def test_the_probe_wait_never_outlives_the_work_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf, "_LAYOUT_PROBE_TIMEOUT_SECONDS", 60.0)
    loop = asyncio.get_running_loop()
    started = loop.time()
    layout = await pdf._read_layout_probe(_Page(None, answers=False), _deadline(0.3))
    assert layout is None
    assert loop.time() - started < 2.0
