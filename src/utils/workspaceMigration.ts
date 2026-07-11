import type { Claim, Denial } from '../../types/billing';

export type LegacyClaim = Omit<Claim, 'workspaceId'> & { workspaceId?: string };
export type LegacyDenial = Omit<Denial, 'workspaceId'> & { workspaceId?: string };

export const migrateClaimWorkspace = (
  claim: LegacyClaim,
  workspaceId: string,
): Claim => ({
  ...claim,
  workspaceId: claim.workspaceId?.trim() || workspaceId,
});

export const migrateDenialWorkspace = (
  denial: LegacyDenial,
  workspaceId: string,
): Denial => ({
  ...denial,
  workspaceId: denial.workspaceId?.trim() || workspaceId,
});
