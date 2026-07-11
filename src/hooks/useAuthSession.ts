import { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';

import type { WorkspaceAccess } from '../../types/access';
import {
  ensureAnonymousSession,
  subscribeToAuth,
} from '../services/auth/authService';
import { ensurePersonalWorkspaceMembership } from '../services/firestore/membershipService';

interface AuthSessionState {
  user: User | null;
  workspaceAccess: WorkspaceAccess | null;
  isLoading: boolean;
  errorMessage: string | null;
}

export function useAuthSession(): AuthSessionState {
  const [user, setUser] = useState<User | null>(null);
  const [workspaceAccess, setWorkspaceAccess] = useState<WorkspaceAccess | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeToAuth((nextUser) => {
      setUser(nextUser);
    });

    void ensureAnonymousSession()
      .then(async (authenticatedUser) => {
        const access = await ensurePersonalWorkspaceMembership(authenticatedUser.uid);
        setWorkspaceAccess(access);
        setIsLoading(false);
      })
      .catch(() => {
        setErrorMessage('Secure Firebase workspace could not be initialized.');
        setIsLoading(false);
      });

    return unsubscribe;
  }, []);

  return { user, workspaceAccess, isLoading, errorMessage };
}
