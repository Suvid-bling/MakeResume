import { describe, expect, it } from 'vitest';
import { countUnits, isCjkText } from '@/lib/print/layout-probe';

describe('layout probe units', () => {
  it('counts words in Latin text and ignores extra spacing', () => {
    expect(countUnits('Cut  latency by 40%', false)).toBe(4);
    expect(countUnits('   ', false)).toBe(0);
  });

  it('counts non-space characters in CJK text', () => {
    expect(countUnits('降低 延迟 40%', true)).toBe(7);
    expect(countUnits('', true)).toBe(0);
  });

  it('treats text as CJK when more than 30% of its characters are CJK', () => {
    expect(isCjkText('将结算延迟降低了百分之四十')).toBe(true);
    expect(isCjkText('Cut latency by 40%')).toBe(false);
    expect(isCjkText('Built the 支付 platform for the team in Berlin')).toBe(false);
    expect(isCjkText('')).toBe(false);
  });
});
