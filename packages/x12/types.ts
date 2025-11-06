/*
 * TypeScript representations of normalized 837/835 payloads.
 * These intentionally mirror the Pydantic models for cross-language parity.
 */

export type Gender = "F" | "M" | "U";
export type Currency = "USD" | "CAD";
export type ServiceUnit = "DAY" | "UNIT" | "VISIT";
export type AdjustmentGroup = "CO" | "PR" | "PI" | "OA";
export type NetworkStatus = "IN" | "OUT";

export interface Contact {
  name: string;
  phone?: string;
  email?: string;
}

export interface Address {
  line1: string;
  line2?: string;
  city: string;
  state: string;
  postal_code: string;
}

export interface Payer {
  id: string;
  name: string;
  contact?: Contact;
}

export interface Provider {
  id: string;
  npi?: string;
  tax_id?: string;
  name: string;
  address?: Address;
  network_status: NetworkStatus;
}

export interface Member {
  id: string;
  first_name: string;
  last_name: string;
  date_of_birth: string;
  gender: Gender;
  address?: Address;
}

export interface Subscriber extends Member {
  relationship: string;
}

export interface DiagnosisCode {
  code: string;
  description?: string;
}

export interface ProcedureCode {
  code: string;
  description?: string;
}

export interface ServiceLine {
  line_number: number;
  procedure_code: ProcedureCode;
  modifiers: string[];
  diagnosis_pointers: number[];
  service_start: string;
  service_end: string;
  charge_amount: number;
  units: number;
  unit_type: ServiceUnit;
  rendering_provider?: Provider;
}

export interface Claim {
  claim_id: string;
  patient: Member;
  subscriber: Subscriber;
  billing_provider: Provider;
  payer: Payer;
  diagnoses: DiagnosisCode[];
  service_lines: ServiceLine[];
  total_charge_amount: number;
  received_at: string;
  currency: Currency;
  control_number?: string;
}

export interface Adjustment {
  group: AdjustmentGroup;
  reason_code: string;
  amount: number;
}

export interface ServiceLinePayment {
  line_number: number;
  paid_amount: number;
  allowed_amount?: number;
  adjustments: Adjustment[];
}

export interface ClaimPayment {
  claim_id: string;
  payment_amount: number;
  patient_responsibility: number;
  adjudication_date: string;
  service_lines: ServiceLinePayment[];
  remark_codes: string[];
}

export interface Remittance {
  remittance_id: string;
  payer: Payer;
  payee: Provider;
  payments: ClaimPayment[];
  payment_issue_date: string;
  check_number?: string;
  currency: Currency;
}

export interface NormalizationMetadata {
  trading_partner: string;
  version: string;
  extracted_at: string;
  source_system: string;
  warnings: string[];
}

export interface NormalizedDocument<TPayload> {
  document_type: "835" | "837";
  metadata: NormalizationMetadata;
  payload: TPayload;
  checksum: string;
}

export interface BatchEnvelope {
  batch_id: string;
  documents: Array<
    | NormalizedDocument<Claim>
    | NormalizedDocument<Remittance>
  >;
  generated_at: string;
}
