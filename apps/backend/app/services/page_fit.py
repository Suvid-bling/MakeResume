"""Render an unsaved resume draft through the real print pipeline to count pages."""

import html
import io
import logging
import re
import statistics
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple
from urllib.parse import urlencode
from uuid import uuid4

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LTTextLine
from pdfminer.pdfpage import PDFPage

from app.ai_budget import AIOperationDeadlineExceeded
from app.config import settings
from app.pdf import PRINT_ERROR_SELECTOR, render_resume_layout, render_resume_pdf
from app.schemas.models import PageFitSettings, TemplateSettings
from app.services.bullet_selector import PageMeasureError

logger = logging.getLogger(__name__)

# A last page filled to at least this share of its height (and at most 100%) counts as
# "full": the 10% tolerance around the page target.
FILL_MIN = 0.90

FillStatus = Literal["fit", "over", "under"]

_MM_TO_PT = 72 / 25.4
# Text whose bottom edge is within this many points of the top margin edge is margin-box
# text (the running header), not content.
_HEADER_TOLERANCE_PT = 1.0


class RenderDraftStore:
    """Short-lived in-memory drafts the print route can fetch by token."""

    def __init__(
        self,
        ttl_seconds: float = 120.0,
        max_entries: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def _evict(self) -> None:
        now = self._clock()
        for token in [t for t, (exp, _) in self._items.items() if exp <= now]:
            del self._items[token]
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def put(self, data: dict[str, Any]) -> str:
        token = uuid4().hex
        self._items[token] = (self._clock() + self._ttl, data)
        self._evict()
        return token

    def get(self, token: str) -> dict[str, Any] | None:
        self._evict()
        item = self._items.get(token)
        return item[1] if item else None

    def discard(self, token: str) -> None:
        self._items.pop(token, None)


render_drafts = RenderDraftStore()


def count_pdf_pages(pdf_bytes: bytes) -> int:
    return len(list(PDFPage.get_pages(io.BytesIO(pdf_bytes))))


def page_limit(fit: PageFitSettings | None, max_pages: int | None) -> int | None:
    """The page cap that applies: None in CV mode, which reports pages and never caps."""
    return None if fit is not None and fit.cv_mode else max_pages


def print_margins(fit: PageFitSettings) -> dict[str, int]:
    return {
        "top": fit.marginTop,
        "right": fit.marginRight,
        "bottom": fit.marginBottom,
        "left": fit.marginLeft,
    }


def resume_print_url(resume_id: str, fit: PageFitSettings) -> str:
    """Print-route URL for a saved resume (same query as GET /resumes/{id}/pdf)."""
    query = urlencode(fit.to_query())
    return f"{settings.frontend_base_url}/print/resumes/{resume_id}?{query}"


async def measure_page_count(data: dict[str, Any], fit: PageFitSettings) -> int:
    """Render ``data`` with the user's print settings and return its page count."""
    token = render_drafts.put(data)
    try:
        query = urlencode(fit.to_query())
        url = f"{settings.frontend_base_url}/print/resumes/draft?draft={token}&{query}"
        margins = print_margins(fit)
        # Also wait for the print route's error marker so a missing draft fails fast.
        pdf_bytes = await render_resume_pdf(
            url,
            fit.pageSize,
            selector=f".resume-print, {PRINT_ERROR_SELECTOR}",
            margins=margins,
        )
        return count_pdf_pages(pdf_bytes)
    except AIOperationDeadlineExceeded:
        raise
    except Exception as e:
        logger.warning("Draft page measurement failed: %s", e)
        raise PageMeasureError("Draft could not be rendered") from e
    finally:
        render_drafts.discard(token)


def page_fit_from_template_settings(
    template_settings: TemplateSettings | dict[str, Any] | None,
    *,
    lang: str | None = None,
    cv_mode: bool = False,
) -> PageFitSettings:
    """Print settings that match a resume's saved formatting.

    ``template_settings`` is a stored settings dict (it may carry extra keys such as
    ``seen``) or a ``TemplateSettings``. A missing or invalid value gives the defaults,
    so a resume that was never formatted prints like the builder's default.
    """
    if template_settings is None:
        return PageFitSettings(lang=lang, cv_mode=cv_mode)
    try:
        saved = (
            template_settings
            if isinstance(template_settings, TemplateSettings)
            else TemplateSettings.model_validate(template_settings)
        )
        return PageFitSettings(
            template=saved.template,
            pageSize=saved.pageSize,
            marginTop=saved.margins.top,
            marginBottom=saved.margins.bottom,
            marginLeft=saved.margins.left,
            marginRight=saved.margins.right,
            sectionSpacing=saved.spacing.section,
            itemSpacing=saved.spacing.item,
            lineHeight=saved.spacing.lineHeight,
            fontSize=saved.fontSize.base,
            headerScale=saved.fontSize.headerScale,
            headerFont=saved.fontSize.headerFont,
            bodyFont=saved.fontSize.bodyFont,
            compactMode=saved.compactMode,
            showContactIcons=saved.showContactIcons,
            accentColor=saved.accentColor,
            descriptionFormat=saved.descriptionFormat,
            lang=lang,
            cv_mode=cv_mode,
        )
    except ValueError as error:
        # pydantic's ValidationError is a ValueError. Keep the details server-side.
        logger.warning("Saved template settings are invalid; using defaults: %s", error)
        return PageFitSettings(lang=lang, cv_mode=cv_mode)


class PdfLayout(NamedTuple):
    """What the PDF itself shows: page count and how full the last page is (0 to 1)."""

    pages: int
    last_page_fill: float


def _text_lines(obj: Any) -> Iterator[LTTextLine]:
    """Every text line of a pdfminer layout object, however deeply nested."""
    if isinstance(obj, LTTextLine):
        yield obj
    elif hasattr(obj, "__iter__"):
        for child in obj:
            yield from _text_lines(child)


def _page_fill(page: LTPage, fit: PageFitSettings) -> float:
    """Share of the printable height, from its top edge down to the lowest text line.

    Text in the top margin (the running header "Name 路 2 / 3", an ``@page`` margin box)
    is not content and is ignored.
    """
    top_margin = fit.marginTop * _MM_TO_PT
    bottom_margin = fit.marginBottom * _MM_TO_PT
    content_top = page.height - top_margin
    content_height = page.height - top_margin - bottom_margin
    if content_height <= 0:
        return 0.0
    lowest: float | None = None
    for line in _text_lines(page):
        if not line.get_text().strip():
            continue
        if line.y0 >= content_top - _HEADER_TOLERANCE_PT:
            continue
        lowest = line.y0 if lowest is None else min(lowest, line.y0)
    if lowest is None:
        return 0.0
    return max(0.0, min(1.0, (content_top - lowest) / content_height))


def measure_pdf_layout(pdf_bytes: bytes, fit: PageFitSettings) -> PdfLayout:
    """Page count and last-page fill of a rendered PDF.

    The fill is how far the lowest content text on the last page reaches down the
    printable area (page height minus the top and bottom margins).
    """
    pages = list(extract_pages(io.BytesIO(pdf_bytes)))
    if not pages:
        return PdfLayout(0, 0.0)
    return PdfLayout(len(pages), _page_fill(pages[-1], fit))


def classify_fill(
    pages: int, last_page_fill: float, target_pages: int | None
) -> FillStatus:
    """Whether a measured resume fits its page target.

    ``fit``: exactly ``target_pages`` pages with the last page filled to ``FILL_MIN`` or
    more. ``over``: more pages than the target. ``under``: fewer pages, or a last page
    emptier than ``FILL_MIN``. No target (unlimited) always fits.
    """
    if target_pages is None:
        return "fit"
    if pages > target_pages:
        return "over"
    if pages < target_pages:
        return "under"
    return "fit" if last_page_fill >= FILL_MIN - 1e-9 else "under"


@dataclass(frozen=True)
class RowLayout:
    """One rendered description row, with the resume path it came from when known."""

    index: int
    style: str
    text: str
    lines: int
    cjk: bool
    units: int
    line_units: tuple[int, ...]
    units_per_line: float
    capacity_known: bool
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "style": self.style,
            "lines": self.lines,
            "cjk": self.cjk,
            "units": self.units,
            "line_units": list(self.line_units),
            "units_per_line": self.units_per_line,
        }


@dataclass(frozen=True)
class LayoutMeasurement:
    """A resume rendered once: the PDF's page count and fill plus the layout probe.

    ``pages`` and ``last_page_fill`` come from the PDF. The ``*_px`` fields and ``rows``
    come from the page's layout probe (zero / empty when it did not answer, see
    ``probe_ok``). ``dom_pages`` is the probe's own page count, for cross-checking.
    """

    pages: int
    last_page_fill: float
    page_height_px: float = 0.0
    content_width_px: float = 0.0
    content_height_px: float = 0.0
    break_waste_px: float = 0.0
    line_height_px: float = 0.0
    dom_pages: int | None = None
    rows: tuple[RowLayout, ...] = field(default_factory=tuple)
    probe_ok: bool = False

    def rows_by_path(self) -> dict[str, RowLayout]:
        """Rows mapped to a resume path (``workExperience[0].description[1]``)."""
        return {row.path: row for row in self.rows if row.path is not None}

    def capacity(self, cjk: bool) -> float | None:
        """Median units (characters for CJK, words otherwise) that fill one line.

        Taken from rows that wrap, since only they show a full line. None when no row
        of that script wraps.
        """
        values = [
            row.units_per_line
            for row in self.rows
            if row.cjk == cjk and row.capacity_known and row.units_per_line > 0
        ]
        return statistics.median(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "last_page_fill": round(self.last_page_fill, 3),
            "page_height_px": round(self.page_height_px, 1),
            "content_width_px": round(self.content_width_px, 1),
            "content_height_px": round(self.content_height_px, 1),
            "break_waste_px": round(self.break_waste_px, 1),
            "line_height_px": round(self.line_height_px, 2),
            "rows": [row.to_dict() for row in self.rows if row.path is not None],
        }


_TAG_RE = re.compile(r"<[^>]*>")
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def row_text_key(value: str) -> str:
    """A description row reduced to what the page shows: no HTML tags or entities, one
    space between words. Rows match the page by this key."""
    plain = html.unescape(_TAG_RE.sub("", _BREAK_RE.sub(" ", value)))
    return " ".join(plain.split())


def description_row_paths(data: dict[str, Any]) -> list[tuple[str, str]]:
    """``(path, text)`` of every description row in the order the data lists them:
    work experience, projects, then the items of custom sections."""
    rows: list[tuple[str, str]] = []

    def add(prefix: str, entries: Any) -> None:
        if not isinstance(entries, list):
            return
        for i, entry in enumerate(entries):
            description = entry.get("description") if isinstance(entry, dict) else None
            if not isinstance(description, list):
                continue
            for j, row in enumerate(description):
                if isinstance(row, str):
                    rows.append((f"{prefix}[{i}].description[{j}]", row))

    add("workExperience", data.get("workExperience"))
    add("personalProjects", data.get("personalProjects"))
    custom = data.get("customSections")
    if isinstance(custom, dict):
        for key, section in custom.items():
            if isinstance(section, dict):
                add(f"customSections.{key}.items", section.get("items"))
    return rows


def map_rows_to_paths(
    rows: list[RowLayout], data: dict[str, Any]
) -> tuple[RowLayout, ...]:
    """Attach each rendered row to the path of the data row with the same text.

    Rows are compared by ``row_text_key`` (HTML and spacing ignored). Equal rows (a
    repeated sentence) take their paths in order: the first rendered copy gets the first
    path. A row with no match (a template that rewrites text) keeps ``path=None``.
    """
    exact: dict[str, deque[str]] = defaultdict(deque)
    folded: dict[str, deque[str]] = defaultdict(deque)
    for path, value in description_row_paths(data):
        key = row_text_key(value)
        if not key:
            continue
        exact[key].append(path)
        folded[key.casefold()].append(path)
    used: set[str] = set()

    def take(queue: deque[str] | None) -> str | None:
        while queue:
            path = queue.popleft()
            if path not in used:
                used.add(path)
                return path
        return None

    mapped: list[RowLayout] = []
    for row in rows:
        key = row_text_key(row.text)
        path = take(exact.get(key)) or take(folded.get(key.casefold()))
        mapped.append(
            RowLayout(
                index=row.index,
                style=row.style,
                text=row.text,
                lines=row.lines,
                cjk=row.cjk,
                units=row.units,
                line_units=row.line_units,
                units_per_line=row.units_per_line,
                capacity_known=row.capacity_known,
                path=path,
            )
        )
    return tuple(mapped)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _parse_probe_row(raw: Any, position: int) -> RowLayout | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        return None
    line_units = raw.get("lineUnits")
    units = (
        tuple(int(_number(v)) for v in line_units)
        if isinstance(line_units, list)
        else ()
    )
    lines = max(1, int(_number(raw.get("lines"), 1)))
    return RowLayout(
        index=int(_number(raw.get("index"), position)),
        style=str(raw.get("style") or "bullet"),
        text=raw["text"],
        lines=lines,
        cjk=bool(raw.get("cjk")),
        units=int(_number(raw.get("units"))),
        line_units=units,
        units_per_line=_number(raw.get("unitsPerLine")),
        capacity_known=bool(raw.get("capacityKnown")),
    )


def merge_layout(
    pdf: PdfLayout, probe: dict[str, Any] | None, data: dict[str, Any]
) -> LayoutMeasurement:
    """Combine the PDF's measurement with the layout probe and map rows to data paths.

    A missing probe gives a measurement with the PDF numbers only (``probe_ok`` False).
    """
    if not isinstance(probe, dict):
        return LayoutMeasurement(pages=pdf.pages, last_page_fill=pdf.last_page_fill)
    raw_rows = probe.get("rows")
    parsed = [
        row
        for position, raw in enumerate(raw_rows if isinstance(raw_rows, list) else [])
        if (row := _parse_probe_row(raw, position)) is not None
    ]
    dom_pages = probe.get("pageCount")
    return LayoutMeasurement(
        pages=pdf.pages,
        last_page_fill=pdf.last_page_fill,
        page_height_px=_number(probe.get("pageHeight")),
        content_width_px=_number(probe.get("contentWidth")),
        content_height_px=_number(probe.get("contentHeight")),
        break_waste_px=_number(probe.get("breakWaste")),
        line_height_px=_number(probe.get("lineHeight")),
        dom_pages=int(dom_pages) if isinstance(dom_pages, (int, float)) else None,
        rows=map_rows_to_paths(parsed, data),
        probe_ok=True,
    )


async def measure_layout(
    data: dict[str, Any], fit: PageFitSettings
) -> LayoutMeasurement:
    """Render ``data`` once with the user's print settings and measure its layout.

    One render gives the PDF's page count and last-page fill, and, through the print
    route's ``probe=1`` mode, each description row's real line count and line capacity.
    """
    token = render_drafts.put(data)
    try:
        query = urlencode({**fit.to_query(), "probe": "1"})
        url = f"{settings.frontend_base_url}/print/resumes/draft?draft={token}&{query}"
        pdf_bytes, probe = await render_resume_layout(
            url,
            fit.pageSize,
            selector=f".resume-print, {PRINT_ERROR_SELECTOR}",
            margins=print_margins(fit),
        )
        measured = merge_layout(measure_pdf_layout(pdf_bytes, fit), probe, data)
        logger.info(
            "Layout measured: pages=%d last_page_fill=%.2f probe_ok=%s rows=%d "
            "mapped=%d format=%s",
            measured.pages,
            measured.last_page_fill,
            measured.probe_ok,
            len(measured.rows),
            len(measured.rows_by_path()),
            fit.descriptionFormat,
        )
        return measured
    except AIOperationDeadlineExceeded:
        raise
    except Exception as e:
        logger.warning("Draft layout measurement failed: %s", e)
        raise PageMeasureError("Draft could not be rendered") from e
    finally:
        render_drafts.discard(token)
