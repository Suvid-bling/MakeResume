/**
 * Layout measurements of a rendered resume, read by the backend (`render_resume_layout`)
 * from `window.__resumeLayout` before it prints the PDF. The backend uses them to plan
 * how much text fits: page height, content height, space lost at page breaks, the
 * line height, and for every description row its real line count and how many
 * characters (CJK) or words (Latin) sit on each line.
 */
import {
  collectKeepZones,
  computePageStarts,
  lineHeightOf,
} from '@/components/preview/page-breaks';
import { getContentAreaPx } from '@/lib/constants/page-dimensions';
import {
  type DescriptionFormat,
  type MarginSettings,
  type PageSize,
} from '@/lib/types/template-settings';
import { textFormatPrefix } from '@/lib/utils/description-styles';

/** Attribute every description row carries (its value is the row style). */
export const DESC_ROW_SELECTOR = '[data-desc-row]';

/** One description row as laid out. */
export interface RowLayout {
  /** Position among `[data-desc-row]` elements in document order. */
  index: number;
  /** Row style: bullet (action), result or plain. */
  style: string;
  /** Plain text of the row without its marker or "- " / "-> " prefix. */
  text: string;
  /** Rendered line count. */
  lines: number;
  /** True when most of the text is CJK, so units are characters instead of words. */
  cjk: boolean;
  /** Characters (CJK) or words (Latin) in the row. */
  units: number;
  /** Units on each rendered line, first to last. */
  lineUnits: number[];
  /** Units that fill one line: the mean of the full lines, or all units for a one-line row. */
  unitsPerLine: number;
  /** True when the row wraps, so `unitsPerLine` comes from full lines (not a lower bound). */
  capacityKnown: boolean;
}

export interface ResumeLayout {
  /** Printable page height in px (page minus top and bottom margins). */
  pageHeight: number;
  /** Printable width in px. */
  contentWidth: number;
  /** Height of the whole resume in px. */
  contentHeight: number;
  /** Pages the break rules produce. */
  pageCount: number;
  /** Start offset of every page in px. */
  pageStarts: number[];
  /** Blank px left at the bottom of pages before the last because of keep-together rules. */
  breakWaste: number;
  /** Used height of the last page divided by `pageHeight`. */
  lastPageFill: number;
  /** Line height of description rows in px. */
  lineHeight: number;
  rows: RowLayout[];
}

export interface ResumeLayoutError {
  error: string;
}

declare global {
  interface Window {
    __resumeLayout?: ResumeLayout | ResumeLayoutError;
  }
}

const CJK_CHAR = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]/;
/** Share of CJK characters above which a row counts in characters, not words. */
const CJK_SHARE = 0.3;

/** Units in a text: non-space characters for CJK, words otherwise. */
export function countUnits(text: string, cjk: boolean): number {
  if (cjk) return Array.from(text.replace(/\s+/g, '')).length;
  return text.split(/\s+/).filter(Boolean).length;
}

export function isCjkText(text: string): boolean {
  const chars = Array.from(text.replace(/\s+/g, ''));
  if (chars.length === 0) return false;
  return chars.filter((char) => CJK_CHAR.test(char)).length / chars.length > CJK_SHARE;
}

/** Text nodes of a row, skipping aria-hidden markers. */
function textNodesOf(row: Element): Text[] {
  const walker = document.createTreeWalker(row, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const hidden = node.parentElement?.closest('[aria-hidden="true"]');
    if (hidden && row.contains(hidden) && hidden !== row) continue;
    nodes.push(node as Text);
  }
  return nodes;
}

/** The text of each rendered line of a row, or null when the browser cannot report it. */
function renderedLines(row: Element, lineHeight: number): string[] | null {
  try {
    const lines: string[] = [];
    let currentTop: number | null = null;
    const range = document.createRange();
    for (const node of textNodesOf(row)) {
      const text = node.data;
      let offset = 0;
      for (const char of text) {
        const length = char.length;
        range.setStart(node, offset);
        range.setEnd(node, offset + length);
        offset += length;
        const rect = range.getClientRects()[0];
        if (!rect || (rect.width === 0 && rect.height === 0)) {
          // Collapsed whitespace has no box: it belongs to the line being built.
          if (lines.length > 0) lines[lines.length - 1] += char;
          continue;
        }
        if (currentTop === null || rect.top > currentTop + lineHeight / 2) {
          currentTop = rect.top;
          lines.push('');
        }
        lines[lines.length - 1] += char;
      }
    }
    const cleaned = lines.map((line) => line.trim()).filter((line) => line.length > 0);
    return cleaned.length > 0 ? cleaned : null;
  } catch {
    return null;
  }
}

function measureRow(row: Element, index: number, textPrefix: boolean): RowLayout {
  const style = row.getAttribute('data-desc-row') || 'bullet';
  const lineHeight = lineHeightOf(row) || 1;
  // In the 'text' format the "- " / "-> " prefix is real text in the first line.
  const marker = textPrefix ? textFormatPrefix(style).trim() : '';
  const stripMarker = (line: string): string =>
    marker && line.startsWith(marker) ? line.slice(marker.length).trim() : line;

  const rendered = renderedLines(row, lineHeight);
  if (!rendered) {
    // No per-line geometry: one line per line-height of the box, units spread evenly.
    const text = stripMarker((row.textContent ?? '').replace(/\s+/g, ' ').trim());
    const lines = Math.max(1, Math.round(row.getBoundingClientRect().height / lineHeight));
    const cjk = isCjkText(text);
    const units = countUnits(text, cjk);
    const perLine = Math.ceil(units / lines);
    return {
      index,
      style,
      text,
      lines,
      cjk,
      units,
      lineUnits: Array.from({ length: lines }, () => perLine),
      unitsPerLine: perLine,
      capacityKnown: lines > 1,
    };
  }
  const lineTexts = [stripMarker(rendered[0]), ...rendered.slice(1)].filter(
    (line, position) => position > 0 || line.length > 0 || rendered.length === 1
  );
  const text = lineTexts.join(' ');
  const cjk = isCjkText(text);
  const lineUnits = lineTexts.map((line) => countUnits(line, cjk));
  const units = lineUnits.reduce((sum, value) => sum + value, 0);
  const full = lineUnits.slice(0, -1);
  const capacityKnown = full.length > 0;
  const unitsPerLine = capacityKnown
    ? full.reduce((sum, value) => sum + value, 0) / full.length
    : units;
  return {
    index,
    style,
    text,
    lines: lineTexts.length,
    cjk,
    units,
    lineUnits,
    unitsPerLine: Math.round(unitsPerLine * 10) / 10,
    capacityKnown,
  };
}
/** Measure the rendered resume inside `container`. */
export function measureResumeLayout(
  container: HTMLElement,
  pageSize: PageSize,
  margins: MarginSettings,
  descriptionFormat: DescriptionFormat = 'bullet'
): ResumeLayout {
  const area = getContentAreaPx(pageSize, margins);
  const pageHeight = area.height;
  const contentHeight = container.scrollHeight;
  const rowElements = Array.from(container.querySelectorAll(DESC_ROW_SELECTOR));
  const rows = rowElements.map((row, index) =>
    measureRow(row, index, descriptionFormat === 'text')
  );
  let pageStarts: number[];
  try {
    pageStarts = computePageStarts(contentHeight, pageHeight, collectKeepZones(container));
  } catch {
    pageStarts = computePageStarts(contentHeight, pageHeight, []);
  }
  let breakWaste = 0;
  for (let i = 0; i < pageStarts.length - 1; i += 1) {
    breakWaste += Math.max(0, pageHeight - (pageStarts[i + 1] - pageStarts[i]));
  }
  const lastStart = pageStarts[pageStarts.length - 1];
  const lastPageFill =
    pageHeight > 0 ? Math.min(1, Math.max(0, (contentHeight - lastStart) / pageHeight)) : 0;
  const lineHeight =
    rowElements.length > 0 ? lineHeightOf(rowElements[0]) : lineHeightOf(container);
  return {
    pageHeight,
    contentWidth: area.width,
    contentHeight,
    pageCount: pageStarts.length,
    pageStarts,
    breakWaste,
    lastPageFill,
    lineHeight,
    rows,
  };
}
