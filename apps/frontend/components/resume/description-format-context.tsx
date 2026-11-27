'use client';

import React, { createContext, useContext } from 'react';
import { type DescriptionFormat, normalizeDescriptionFormat } from '@/lib/types/template-settings';

/**
 * The resume's description format, provided by <Resume> to every template.
 *
 * Lives in a client module so the server-rendered print route and the client preview
 * share one context; the default is 'bullet', so a template rendered outside <Resume>
 * looks as it always did.
 */
export const DescriptionFormatContext = createContext<DescriptionFormat>('bullet');

interface DescriptionFormatProviderProps {
  format?: DescriptionFormat;
  children: React.ReactNode;
}

export const DescriptionFormatProvider: React.FC<DescriptionFormatProviderProps> = ({
  format,
  children,
}) => (
  <DescriptionFormatContext.Provider value={normalizeDescriptionFormat(format)}>
    {children}
  </DescriptionFormatContext.Provider>
);

export function useDescriptionFormat(): DescriptionFormat {
  return useContext(DescriptionFormatContext);
}
