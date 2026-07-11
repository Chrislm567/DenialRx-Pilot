import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  query,
  setDoc,
  where,
} from 'firebase/firestore';

import type { Claim } from '../../../types/billing';
import { firestoreDb } from '../../lib/firebase';
import { FIRESTORE_COLLECTIONS } from './collectionNames';
import { claimConverter } from './converters';

const claimsCollection = collection(
  firestoreDb,
  FIRESTORE_COLLECTIONS.claims,
).withConverter(claimConverter);

export const listClaims = async (workspaceId: string): Promise<Claim[]> => {
  const workspaceQuery = query(claimsCollection, where('workspaceId', '==', workspaceId));
  const snapshot = await getDocs(workspaceQuery);
  return snapshot.docs.map((claimDocument) => claimDocument.data());
};

export const getClaimById = async (
  claimId: string,
  workspaceId: string,
): Promise<Claim | null> => {
  const snapshot = await getDoc(doc(claimsCollection, claimId));
  const claim = snapshot.exists() ? snapshot.data() : null;
  return claim?.workspaceId === workspaceId ? claim : null;
};

export const saveClaim = async (claim: Claim): Promise<void> => {
  await setDoc(doc(claimsCollection, claim.id), claim);
};

export const updateClaim = async (
  claimId: string,
  workspaceId: string,
  updates: Partial<Omit<Claim, 'id' | 'workspaceId'>>,
): Promise<void> => {
  await setDoc(
    doc(claimsCollection, claimId),
    { id: claimId, workspaceId, ...updates } as Claim,
    { merge: true },
  );
};

export const removeClaim = async (claimId: string): Promise<void> => {
  await deleteDoc(doc(claimsCollection, claimId));
};
