export type SupportedCarcCode = '197' | '50' | '29';

export interface ParsedRemittanceFields {
  claimId?: string;
  carcCode?: SupportedCarcCode;
}

const SUPPORTED_CARC_CODES = new Set<SupportedCarcCode>(['197', '50', '29']);
const carcPattern = /\bCO[-\s]?(197|50|29)\b/i;
const remittanceNumberPattern = /\bREMIT\s+NO:\s*([0-9]+)\b/i;

export function parseRemittanceText(rawText: string): ParsedRemittanceFields {
  const carcMatch = rawText.match(carcPattern);
  const remitMatch = rawText.match(remittanceNumberPattern);
  const matchedCarc = carcMatch?.[1] as SupportedCarcCode | undefined;

  return {
    claimId: remitMatch?.[1] ? `CLM-${remitMatch[1]}` : undefined,
    carcCode:
      matchedCarc && SUPPORTED_CARC_CODES.has(matchedCarc) ? matchedCarc : undefined,
  };
}
