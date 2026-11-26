import type { Metadata, Viewport } from 'next';
import localFont from 'next/font/local';
import { THEME_COLOR } from '@/lib/theme-color';
import './(default)/css/globals.css';

// Original Google Fonts files are bundled locally so builds need no font downloads.
const spaceGrotesk = localFont({
  src: './fonts/SpaceGrotesk-Variable.ttf',
  weight: '300 700',
  variable: '--font-space-grotesk',
  display: 'swap',
});

// Lato for the Modern Deedy and Deedy resume templates. Each weight/style is its
// own file; `preload: false` keeps them off pages that do not use those templates.
const lato = localFont({
  src: [
    { path: './fonts/Lato-Thin.ttf', weight: '100', style: 'normal' },
    { path: './fonts/Lato-ThinItalic.ttf', weight: '100', style: 'italic' },
    { path: './fonts/Lato-Light.ttf', weight: '300', style: 'normal' },
    { path: './fonts/Lato-LightItalic.ttf', weight: '300', style: 'italic' },
    { path: './fonts/Lato-Regular.ttf', weight: '400', style: 'normal' },
    { path: './fonts/Lato-Italic.ttf', weight: '400', style: 'italic' },
    { path: './fonts/Lato-Bold.ttf', weight: '700', style: 'normal' },
    { path: './fonts/Lato-BoldItalic.ttf', weight: '700', style: 'italic' },
  ],
  variable: '--font-lato',
  display: 'swap',
  preload: false,
});

// Raleway for the Deedy template's contact lines, roles and dates.
const raleway = localFont({
  src: './fonts/Raleway-Variable.ttf',
  weight: '100 900',
  variable: '--font-raleway',
  display: 'swap',
  preload: false,
});

// Complete CJK families preserve the original glyphs for every locale without
// Google's per-subset downloads. The per-locale ordering lives in
// lib/types/template-settings.ts.
const notoSansSC = localFont({
  src: './fonts/NotoSansSC-Variable.ttf',
  weight: '100 900',
  variable: '--font-noto-sans-sc',
  display: 'swap',
  preload: false,
});

const notoSansKR = localFont({
  src: './fonts/NotoSansKR-Variable.ttf',
  weight: '100 900',
  variable: '--font-noto-sans-kr',
  display: 'swap',
  preload: false,
});

const notoSansJP = localFont({
  src: './fonts/NotoSansJP-Variable.ttf',
  weight: '100 900',
  variable: '--font-noto-sans-jp',
  display: 'swap',
  preload: false,
});

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
      <body
        className={`${spaceGrotesk.variable} ${lato.variable} ${raleway.variable} ${notoSansSC.variable} ${notoSansKR.variable} ${notoSansJP.variable} antialiased bg-canvas text-ink-soft min-h-full`}
      >
        {children}
      </body>
    </html>
  );
}
