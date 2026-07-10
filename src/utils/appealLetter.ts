import { denialDatabase } from '../../constants/denialDatabase';
import type { AppealTemplate, Claim } from '../../types/billing';

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
});

const safeText = (value: string, fallback: string): string => {
  const normalized = value.trim();
  return normalized || fallback;
};

const safeBalance = (value: number): string =>
  Number.isFinite(value) && value > 0
    ? currencyFormatter.format(value)
    : '[MISSING BALANCE DATA]';

export interface AppealLetterResult {
  template: AppealTemplate | null;
  letterText: string;
  missingFields: string[];
}

export function renderAppealLetter(
  claim: Claim,
  carcCode: string,
): AppealLetterResult {
  const template = denialDatabase.find((item) => item.carcCode === carcCode) ?? null;
  const missingFields = [
    !claim.id.trim() ? 'Claim ID' : null,
    !claim.insurancePayer.trim() ? 'Insurance Payer' : null,
    !Number.isFinite(claim.totalBilled) || claim.totalBilled <= 0 ? 'Total Balance' : null,
  ].filter((field): field is string => field !== null);

  if (!template) {
    return {
      template: null,
      missingFields,
      letterText: `No appeal template is available for CARC ${carcCode || '[MISSING CARC DATA]'}.`,
    };
  }

  const replacements: Record<string, string> = {
    claimId: safeText(claim.id, '[MISSING CLAIM DATA]'),
    balance: safeBalance(claim.totalBilled),
    payer: safeText(claim.insurancePayer, '[MISSING PROVIDER DATA]'),
  };

  const letterText = template.standardLetterBody.replace(
    /{{(claimId|balance|payer)}}/g,
    (_, key: keyof typeof replacements) => replacements[key],
  );

  return { template, letterText, missingFields };
}
