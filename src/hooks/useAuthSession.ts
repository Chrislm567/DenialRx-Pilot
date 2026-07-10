import { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';

import {
  ensureAnonymousSession,
  subscribeToAuth,
} from '../services/auth/authService';

interface AuthSessionState {
  user: User | null;
  isLoading: boolean;
  errorMessage: string | null;
}

export function useAuthSession(): AuthSessionState {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeToAuth((nextUser) => {
      setUser(nextUser);
      if (nextUser) setIsLoading(false);
    });

    void ensureAnonymousSession().catch(() => {
      setErrorMessage('Secure Firebase session could not be initialized.');
      setIsLoading(false);
    });

    return unsubscribe;
  }, []);

  return { user, isLoading, errorMessage };
}
