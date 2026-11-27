import type { Metadata, Viewport } from 'next';
import { THEME_COLOR } from '@/lib/theme-color';

// Web fonts are self-hosted from npm (@fontsource), so neither `next dev` nor
// `next build` needs to reach Google Fonts. Every sheet declares
// `font-display: swap` and splits the face by unicode-range, so the browser
// only downloads the ranges a page actually renders. The CSS variables these
// faces are exposed through live in ./fonts.css.

// UI face (globals.css --font-mono). Variable font: one file per subset covers 300-700.
import '@fontsource-variable/space-grotesk';

// Lato for the Modern Deedy and Deedy resume templates. A static family, so each
// weight/style is its own sheet. Deedy draws the first name in Hairline (100).
import '@fontsource/lato/100.css';
import '@fontsource/lato/100-italic.css';
import '@fontsource/lato/300.css';
import '@fontsource/lato/300-italic.css';
import '@fontsource/lato/400.css';
import '@fontsource/lato/400-italic.css';
import '@fontsource/lato/700.css';
import '@fontsource/lato/700-italic.css';

// Raleway for the Deedy template's contact lines, roles and dates. Variable font.
import '@fontsource-variable/raleway';

// CJK fallbacks for Chinese/Japanese/Korean resume content. All three regional
// faces are loaded because Noto Sans SC covers only 0.7% of Hangul (Korean
// rendered as tofu) while covering ~93% of kana (hijacking Japanese with
// Simplified-Chinese glyph forms). The per-locale ordering lives in
// lib/types/template-settings.ts.
import '@fontsource-variable/noto-sans-sc';
import '@fontsource-variable/noto-sans-kr';
import '@fontsource-variable/noto-sans-jp';

import './fonts.css';
import './(default)/css/globals.css';

export const metadata: Metadata = {
  title: 'Resume Matcher',
  description: 'Build your resume with Resume Matcher',
  applicationName: 'Resume Matcher',
  keywords: ['resume', 'matcher', 'job', 'application'],
};

export const viewport: Viewport = { themeColor: THEME_COLOR };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-US" className="h-full" suppressHydrationWarning>
      <body className="antialiased bg-canvas text-ink-soft min-h-full">{children}</body>
    </html>
  );
}
