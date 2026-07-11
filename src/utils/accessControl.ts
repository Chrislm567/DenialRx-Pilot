import type { WorkspaceRole } from '../../types/access';

export type WorkspaceAction = 'read' | 'write' | 'manageMembers';

const rolePermissions: Record<WorkspaceRole, ReadonlySet<WorkspaceAction>> = {
  admin: new Set(['read', 'write', 'manageMembers']),
  biller: new Set(['read', 'write']),
  viewer: new Set(['read']),
};

export const canPerformWorkspaceAction = (
  role: WorkspaceRole,
  action: WorkspaceAction,
): boolean => rolePermissions[role].has(action);
