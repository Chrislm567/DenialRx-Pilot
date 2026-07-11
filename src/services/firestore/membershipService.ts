import { doc, getDoc, setDoc } from 'firebase/firestore';

import type { WorkspaceAccess, WorkspaceMembership } from '../../../types/access';
import { firestoreDb } from '../../lib/firebase';

const membershipRef = (workspaceId: string, userId: string) =>
  doc(firestoreDb, 'workspaces', workspaceId, 'members', userId);

export const ensurePersonalWorkspaceMembership = async (
  userId: string,
): Promise<WorkspaceAccess> => {
  const reference = membershipRef(userId, userId);
  const snapshot = await getDoc(reference);

  if (!snapshot.exists()) {
    const membership: WorkspaceMembership = {
      workspaceId: userId,
      userId,
      role: 'admin',
      createdAt: new Date().toISOString(),
    };
    await setDoc(reference, membership);
    return { workspaceId: userId, role: 'admin' };
  }

  const membership = snapshot.data() as WorkspaceMembership;
  return { workspaceId: membership.workspaceId, role: membership.role };
};
