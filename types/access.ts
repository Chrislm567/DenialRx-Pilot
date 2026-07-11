export type WorkspaceRole = 'admin' | 'biller' | 'viewer';

export interface WorkspaceMembership {
  workspaceId: string;
  userId: string;
  role: WorkspaceRole;
  createdAt: string;
}

export interface WorkspaceAccess {
  workspaceId: string;
  role: WorkspaceRole;
}
