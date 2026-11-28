import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DescriptionList, DescriptionRowText } from '@/components/resume/description-list';
import { DescriptionFormatProvider } from '@/components/resume/description-format-context';
import {
  normalizeDescriptionStyle,
  rowHasMarker,
  textFormatPrefix,
} from '@/lib/utils/description-styles';
import {
  DEFAULT_TEMPLATE_SETTINGS,
  normalizeDescriptionFormat,
} from '@/lib/types/template-settings';
import { getResumePdfUrl, toPageFitSettings } from '@/lib/api/resume';
import { buildPreviewLengthOptions } from '@/lib/utils/tailor-length-settings';

describe('description row style helpers', () => {
  it('maps unknown or missing values to a bullet', () => {
    expect(normalizeDescriptionStyle(undefined)).toBe('bullet');
    expect(normalizeDescriptionStyle('weird')).toBe('bullet');
    expect(normalizeDescriptionStyle('result')).toBe('result');
    expect(normalizeDescriptionStyle('plain')).toBe('plain');
  });

  it('gives a marker to bullet and result rows but never to plain rows', () => {
    expect(rowHasMarker('bullet')).toBe(true);
    expect(rowHasMarker('result')).toBe(true);
    expect(rowHasMarker('plain')).toBe(false);
    expect(rowHasMarker(undefined)).toBe(true);
  });

  it('writes the text-format prefix as "- ", "-> " or nothing', () => {
    expect(textFormatPrefix('bullet')).toBe('- ');
    expect(textFormatPrefix('result')).toBe('-> ');
    expect(textFormatPrefix('plain')).toBe('');
    expect(textFormatPrefix(undefined)).toBe('- ');
  });

  it('falls back to bullets for an unknown description format', () => {
    expect(normalizeDescriptionFormat('text')).toBe('text');
    expect(normalizeDescriptionFormat('bullet')).toBe('bullet');
    expect(normalizeDescriptionFormat('markdown')).toBe('bullet');
    expect(normalizeDescriptionFormat(undefined)).toBe('bullet');
  });
});

describe('DescriptionList in the text format', () => {
  const items = ['Cut latency', 'Latency fell 40%', 'Team overview'];
  const styles = ['bullet', 'result', 'plain'] as const;

  it('prints the prefix as text, with no marker span and no hanging indent', () => {
    const { container } = render(
      <DescriptionFormatProvider format="text">
        <DescriptionList items={items} styles={[...styles]} />
      </DescriptionFormatProvider>
    );
    const rows = screen.getAllByRole('listitem');
    expect(rows.map((row) => row.textContent)).toEqual([
      '- Cut latency',
      '-> Latency fell 40%',
      'Team overview',
    ]);
    for (const row of rows) expect(row).not.toHaveClass('ml-4', 'flex');
    expect(container.querySelector('[aria-hidden="true"]')).toBeNull();
  });

  it('tags every row with its style for the layout probe', () => {
    render(
      <DescriptionFormatProvider format="text">
        <DescriptionList items={items} styles={[...styles]} />
      </DescriptionFormatProvider>
    );
    expect(screen.getAllByRole('listitem').map((row) => row.dataset.descRow)).toEqual([
      'bullet',
      'result',
      'plain',
    ]);
  });

  it('keeps the marker layout in the bullet format, result rows included', () => {
    render(<DescriptionList items={items} styles={[...styles]} />);
    const rows = screen.getAllByRole('listitem');
    expect(rows[0]).toHaveClass('ml-4');
    expect(rows[1]).toHaveClass('ml-4'); // a result row shows a marker too
    expect(rows[2]).not.toHaveClass('ml-4');
    expect(rows.map((row) => row.dataset.descRow)).toEqual(['bullet', 'result', 'plain']);
    expect(rows[1].textContent).not.toContain('->');
  });
});

describe('DescriptionRowText', () => {
  it('adds the prefix only in the text format', () => {
    const { container, rerender } = render(
      <DescriptionRowText html="Did a thing" style="result" />
    );
    expect(container.textContent).toBe('Did a thing');
    rerender(
      <DescriptionFormatProvider format="text">
        <DescriptionRowText html="Did a thing" style="result" />
      </DescriptionFormatProvider>
    );
    expect(container.textContent).toBe('-> Did a thing');
  });
});

describe('description format plumbing', () => {
  const text = { ...DEFAULT_TEMPLATE_SETTINGS, descriptionFormat: 'text' as const };

  it('defaults a resume to the bullet format', () => {
    expect(DEFAULT_TEMPLATE_SETTINGS.descriptionFormat).toBe('bullet');
    expect(toPageFitSettings(DEFAULT_TEMPLATE_SETTINGS).descriptionFormat).toBe('bullet');
  });

  it('carries the format into page-fit settings and the PDF URL', () => {
    expect(toPageFitSettings(text).descriptionFormat).toBe('text');
    const url = new URL(getResumePdfUrl('abc', text), 'http://localhost');
    expect(url.searchParams.get('descriptionFormat')).toBe('text');
    const bullet = new URL(getResumePdfUrl('abc', DEFAULT_TEMPLATE_SETTINGS), 'http://localhost');
    expect(bullet.searchParams.get('descriptionFormat')).toBe('bullet');
  });

  it('tells the backend about the text format even with unlimited pages', () => {
    const options = buildPreviewLengthOptions(
      { maxPages: null, maxBulletsPerEntry: null },
      toPageFitSettings(text)
    );
    expect(options).toEqual({ descriptionFormat: 'text' });
  });
});
