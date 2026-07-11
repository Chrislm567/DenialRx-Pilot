export type ClaimStatus = 'New' | 'In Progress' | 'Appealed' | 'Paid';

export interface WorkspaceOwned {
  workspaceId: string;
}

export interface Claim extends WorkspaceOwned {
  id: string;
  mockPatientName: string;
  insurancePayer: string;
  dateOfService: string;
  totalBilled: number;
  status: ClaimStatus;
}

export interface Denial extends WorkspaceOwned {
  id: string;
  claimId: string;
  carcCode: string;
  rarcCode: string;
  dateDenied: string;
  description: string;
}

export interface AppealTemplate {
  carcCode: string;
  denialReason: string;
  argumentStrategy: string;
  standardLetterBody: string;
}
