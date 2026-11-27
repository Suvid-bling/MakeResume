'use client';

import { useEffect } from 'react';
import { measureResumeLayout } from '@/lib/print/layout-probe';
import {
  type DescriptionFormat,
  type MarginSettings,
  type PageSize,
} from '@/lib/types/template-settings';

interface LayoutProbeProps {
  pageSize: PageSize;
  /** The real page margins in mm (the print route zeroes them in CSS; Playwright applies them). */
  margins: MarginSettings;
  descriptionFormat: DescriptionFormat;
}

/**
 * Print-route helper, mounted only for `?probe=1`. Once fonts are loaded it measures the
 * rendered resume and publishes the result as `window.__resumeLayout`, which the backend
 * reads before it prints the PDF. It renders nothing.
 */
export function LayoutProbe({ pageSize, margins, descriptionFormat }: LayoutProbeProps) {
  const { top, bottom, left, right } = margins;
  useEffect(() => {
    let cancelled = false;
    const publish = () => {
      if (cancelled) return;
      try {
        const container = document.querySelector<HTMLElement>('.resume-print');
        if (!container) {
          window.__resumeLayout = { error: 'resume-not-found' };
          return;
        }
        window.__resumeLayout = measureResumeLayout(
          container,
          pageSize,
          { top, bottom, left, right },
          descriptionFormat
        );
      } catch (error) {
        console.error('Layout probe failed:', error);
        window.__resumeLayout = { error: 'probe-failed' };
      }
    };
    void document.fonts.ready.then(() => {
      // One frame after fonts so the final line breaks are laid out.
      requestAnimationFrame(() => requestAnimationFrame(publish));
    });
    return () => {
      cancelled = true;
    };
  }, [pageSize, top, bottom, left, right, descriptionFormat]);
  return null;
}
