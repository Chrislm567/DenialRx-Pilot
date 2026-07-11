import { describe, expect, it } from 'vitest';

import type { Claim } from '../../types/billing';
import { renderAppealLetter } from './appealLetter';

const baseClaim: Claim = {
  id: 'CLM-TEST-1',
  workspaceId: 'mock-test-workspace',
  mockPatientName: 'Mock Patient Test',
  insurancePayer: 'Mock Test Payer',
  dateOfService: '2026-07-10',
  totalBilled: 1250,
  status: 'New',
};

describe('renderAppealLetter', () => {
  it('merges claim values into the matching CARC template', () => {
    const result = renderAppealLetter(baseClaim, '197');

    expect(result.template?.carcCode).toBe('197');
    expect(result.letterText).toContain('CLM-TEST-1');
    expect(result.letterText).toContain('$1,250.00');
    expect(result.letterText).toContain('Mock Test Payer');
    expect(result.missingFields).toEqual([]);
  });

  it('uses safe markers for missing merge data', () => {
    const result = renderAppealLetter(
      { ...baseClaim, id: '', insurancePayer: '', totalBilled: 0 },
      '50',
    );

    expect(result.letterText).toContain('[MISSING CLAIM DATA]');
    expect(result.letterText).toContain('[MISSING PROVIDER DATA]');
    expect(result.letterText).toContain('[MISSING BALANCE DATA]');
    expect(result.missingFields).toEqual(['Claim ID', 'Insurance Payer', 'Total Balance']);
  });
});
