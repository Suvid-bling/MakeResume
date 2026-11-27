'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import {
  normalizeDescriptionStyle,
  rowHasMarker,
  textFormatPrefix,
  type DescriptionStyle,
} from '@/lib/utils/description-styles';
import { useDescriptionFormat } from './description-format-context';
import { SafeHtml } from './safe-html';
import baseStyles from './styles/_base.module.css';

export type { DescriptionStyle };

interface DescriptionListProps {
  items?: string[];
  styles?: DescriptionStyle[];
  textClassName?: string;
  marker?: string;
  markerClassName?: string;
}

export const DescriptionList: React.FC<DescriptionListProps> = ({
  items,
  styles,
  textClassName = baseStyles['resume-text-sm'],
  marker = '\u2022',
  markerClassName = 'mr-1.5 flex-shrink-0',
}) => {
  const format = useDescriptionFormat();
  if (!items || items.length === 0) return null;

  // Text format: the "- " / "-> " prefix is real text inside the line. No marker
  // span and no hanging indent, so a wrapped line returns to the left edge.
  if (format === 'text') {
    return (
      <ul className={cn(baseStyles['resume-list'], textClassName)}>
        {items.map((desc, index) => (
          <li key={index} data-desc-row={normalizeDescriptionStyle(styles?.[index])}>
            {textFormatPrefix(styles?.[index])}
            <SafeHtml html={desc} />
          </li>
        ))}
      </ul>
    );
  }

  return (
    <ul className={cn(baseStyles['resume-list'], textClassName)}>
      {items.map((desc, index) => {
        const showMarker = rowHasMarker(styles?.[index]);
        return (
          <li
            key={index}
            data-desc-row={normalizeDescriptionStyle(styles?.[index])}
            className={cn('flex', showMarker && 'ml-4')}
          >
            {showMarker && (
              <span className={markerClassName} aria-hidden="true">
                {marker}
                {'\u00A0'}
              </span>
            )}
            <span>
              <SafeHtml html={desc} />
            </span>
          </li>
        );
      })}
    </ul>
  );
};

interface DescriptionRowTextProps {
  html: string;
  style?: DescriptionStyle;
}

/**
 * One description line for templates that lay rows out themselves (no marker). In the
 * 'text' format it carries the "- " / "-> " prefix; in 'bullet' format it is the text alone.
 */
export const DescriptionRowText: React.FC<DescriptionRowTextProps> = ({ html, style }) => {
  const format = useDescriptionFormat();
  return (
    <>
      {format === 'text' ? textFormatPrefix(style) : ''}
      <SafeHtml html={html} />
    </>
  );
};
