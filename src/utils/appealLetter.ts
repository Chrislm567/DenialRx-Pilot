import { denialDatabase } from '../../constants/denialDatabase';
import type { AppealTemplate, Claim } from '../../types/billing';

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
});

export interface AppealLetterResult {
  template: AppealTemplate | null;
  letterText: string;
}

export function renderAppealLetter(
  claim: Claim,
  carcCode: string,
): AppealLetterResult {
  const template = denialDatabase.find((item) => item.carcCode === carcCode) ?? null;

  if (!template) {
    return {
      template: null,
      letterText: `No appeal template is available for CARC ${carcCode}.`,
    };
  }

  const replacements: Record<string, string> = {
    claimId: claim.id,
    balance: currencyFormatter.format(claim.totalBilled),
    payer: claim.insurancePayer,
  };

  const letterText = template.standardLetterBody.replace(
    /{{(claimId|balance|payer)}}/g,
    (_, key: keyof typeof replacements) => replacements[key],
  );

  return { template, letterText };
}