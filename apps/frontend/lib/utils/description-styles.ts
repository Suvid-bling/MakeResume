/**
 * Row style of one description point. `bullet` is an action row, `result` an
 * outcome row, `plain` a row without a marker. This is the single source of the
 * type for the whole frontend.
 */
export type DescriptionStyle = 'bullet' | 'result' | 'plain';

/** Editor toggle order: bullet -> result -> plain -> bullet. */
const STYLE_CYCLE: readonly DescriptionStyle[] = ['bullet', 'result', 'plain'];

/** Map any stored value to a supported style. Unknown or missing values become `bullet`. */
export function normalizeDescriptionStyle(value: unknown): DescriptionStyle {
  return value === 'plain' || value === 'result' ? value : 'bullet';
}

/** Bullet and result rows carry a marker in bullet format; plain rows never do. */
export function rowHasMarker(style: unknown): boolean {
  return normalizeDescriptionStyle(style) !== 'plain';
}

/**
 * The literal text a row starts with in the 'text' description format: "- " for an
 * action row, "-> " for a result row, nothing for a plain row.
 */
export function textFormatPrefix(style: unknown): string {
  const normalized = normalizeDescriptionStyle(style);
  if (normalized === 'plain') return '';
  return normalized === 'result' ? '-> ' : '- ';
}
/** The style the per-row toggle switches to next. */
export function nextDescriptionStyle(style: DescriptionStyle): DescriptionStyle {
  return STYLE_CYCLE[(STYLE_CYCLE.indexOf(style) + 1) % STYLE_CYCLE.length];
}

export function alignDescriptionStyles(
  descriptions: string[] | undefined,
  styles: (DescriptionStyle | null | undefined)[] | undefined
): DescriptionStyle[] {
  return (descriptions || []).map((_, index) => normalizeDescriptionStyle(styles?.[index]));
}

/** A description point paired with its style, so a reorder moves both together. */
export interface DescriptionRow {
  /** Drag-and-drop id (the point's position when the rows were built). */
  id: number;
  text: string;
  style: DescriptionStyle;
}

export function toDescriptionRows(
  descriptions: string[] | undefined,
  styles: (DescriptionStyle | null | undefined)[] | undefined
): DescriptionRow[] {
  const alignedStyles = alignDescriptionStyles(descriptions, styles);
  return (descriptions || []).map((text, index) => ({
    id: index,
    text,
    style: alignedStyles[index],
  }));
}

export function fromDescriptionRows(rows: DescriptionRow[]): {
  description: string[];
  descriptionStyles: DescriptionStyle[];
} {
  return {
    description: rows.map((row) => row.text),
    descriptionStyles: rows.map((row) => row.style),
  };
}

export function toggleDescriptionStyle(
  descriptions: string[] | undefined,
  styles: (DescriptionStyle | null | undefined)[] | undefined,
  index: number
): DescriptionStyle[] {
  const alignedStyles = alignDescriptionStyles(descriptions, styles);

  if (index < 0 || index >= alignedStyles.length) {
    return alignedStyles;
  }

  alignedStyles[index] = nextDescriptionStyle(alignedStyles[index]);
  return alignedStyles;
}
