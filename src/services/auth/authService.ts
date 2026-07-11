import { onAuthStateChanged, signInAnonymously, type User } from 'firebase/auth';

import { firebaseAuth } from '../../lib/firebase';

export const subscribeToAuth = (callback: (user: User | null) => void): (() => void) =>
  onAuthStateChanged(firebaseAuth, callback);

export const ensureAnonymousSession = async (): Promise<User> => {
  if (firebaseAuth.currentUser) return firebaseAuth.currentUser;

  const credential = await signInAnonymously(firebaseAuth);
  return credential.user;
};

export const getCurrentWorkspaceId = (): string => {
  const workspaceId = firebaseAuth.currentUser?.uid;

  if (!workspaceId) {
    throw new Error('An authenticated DenialRx workspace session is required.');
  }

  return workspaceId;
};
