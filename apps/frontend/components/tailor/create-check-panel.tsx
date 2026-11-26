'use client';

import { Check, Info, Warning } from '@phosphor-icons/react';
import type { CreateCheckItem } from '@/components/common/resume_previewer_context';
import { useTranslations } from '@/lib/i18n';
import { cn } from '@/lib/utils';

interface CreateCheckPanelProps {
  checks: CreateCheckItem[];
}

const ICONS: Record<CreateCheckItem['severity'], typeof Check> = {
  info: Info,
  warning: Warning,
  error: Warning,
};

const ICON_TONE: Record<CreateCheckItem['severity'], string> = {
  info: 'text-ink-soft',
  warning: 'text-warning-text',
  error: 'text-destructive',
};

/** Style/grounding check results for the "create" tailor mode's generated content. */
export function CreateCheckPanel({ checks }: CreateCheckPanelProps) {
  const { t } = useTranslations();

  const translateCode = (code: string) => {
    const key = `tailor.createChecks.codes.${code}`;
    const translated = t(key);
    return translated === key ? code : translated;
  };

  return (
    <section
      aria-label={t('tailor.createChecks.title')}
      className="space-y-2 border border-ink bg-white p-4 shadow-sw-nested"
    >
      <h3 className="font-mono text-xs font-bold uppercase tracking-wider text-ink">
        {t('tailor.createChecks.title')}
      </h3>
      {checks.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-success">
          <Check aria-hidden="true" className="size-4 shrink-0" />
          {t('tailor.createChecks.allClear')}
        </p>
      ) : (
        <ul className="space-y-2">
          {checks.map((check, index) => {
            const Icon = ICONS[check.severity] ?? Info;
            return (
              <li key={`${check.code}-${index}`} className="flex items-start gap-2 text-sm">
                <Icon
                  aria-hidden="true"
                  className={cn('size-4 shrink-0', ICON_TONE[check.severity])}
                />
                <span>
                  <span className="font-bold">{translateCode(check.code)}</span>
                  {': '}
                  {check.detail}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
