import { describe, expect, it } from 'vitest';

import { parseRemittanceText } from './textParser';

describe('parseRemittanceText', () => {
  it('extracts a supported CARC code and remittance number', () => {
    expect(parseRemittanceText('REMIT NO: 20481 adjustment CO-197')).toEqual({
      claimId: 'CLM-20481',
      carcCode: '197',
    });
  });

  it('returns undefined fields when markers are absent', () => {
    expect(parseRemittanceText('No supported remittance markers')).toEqual({
      claimId: undefined,
      carcCode: undefined,
    });
  });
});
