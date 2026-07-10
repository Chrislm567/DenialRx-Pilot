import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  setDoc,
} from 'firebase/firestore';

import type { Claim } from '../../../types/billing';
import { firestoreDb } from '../../lib/firebase';
import { FIRESTORE_COLLECTIONS } from './collectionNames';
import { claimConverter } from './converters';

const claimsCollection = collection(
  firestoreDb,
  FIRESTORE_COLLECTIONS.claims,
).withConverter(claimConverter);

export const listClaims = async (): Promise<Claim[]> => {
  const snapshot = await getDocs(claimsCollection);
  return snapshot.docs.map((claimDocument) => claimDocument.data());
};

export const getClaimById = async (claimId: string): Promise<Claim | null> => {
  const snapshot = await getDoc(doc(claimsCollection, claimId));
  return snapshot.exists() ? snapshot.data() : null;
};

export const saveClaim = async (claim: Claim): Promise<void> => {
  await setDoc(doc(claimsCollection, claim.id), claim);
};

export const updateClaim = async (
  claimId: string,
  updates: Partial<Omit<Claim, 'id'>>,
): Promise<void> => {
  await setDoc(
    doc(claimsCollection, claimId),
    { id: claimId, ...updates } as Claim,
    { merge: true },
  );
};

export const removeClaim = async (claimId: string): Promise<void> => {
  await deleteDoc(doc(claimsCollection, claimId));
};
