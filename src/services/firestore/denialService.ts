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

import type { Denial } from '../../../types/billing';
import { firestoreDb } from '../../lib/firebase';
import { FIRESTORE_COLLECTIONS } from './collectionNames';
import { denialConverter } from './converters';

const denialsCollection = collection(
  firestoreDb,
  FIRESTORE_COLLECTIONS.denials,
).withConverter(denialConverter);

export const listDenials = async (): Promise<Denial[]> => {
  const snapshot = await getDocs(denialsCollection);
  return snapshot.docs.map((denialDocument) => denialDocument.data());
};

export const listDenialsByClaimId = async (claimId: string): Promise<Denial[]> => {
  const denialQuery = query(denialsCollection, where('claimId', '==', claimId));
  const snapshot = await getDocs(denialQuery);
  return snapshot.docs.map((denialDocument) => denialDocument.data());
};

export const getDenialById = async (denialId: string): Promise<Denial | null> => {
  const snapshot = await getDoc(doc(denialsCollection, denialId));
  return snapshot.exists() ? snapshot.data() : null;
};

export const saveDenial = async (denial: Denial): Promise<void> => {
  await setDoc(doc(denialsCollection, denial.id), denial);
};

export const updateDenial = async (
  denialId: string,
  updates: Partial<Omit<Denial, 'id'>>,
): Promise<void> => {
  await setDoc(
    doc(denialsCollection, denialId),
    { id: denialId, ...updates } as Denial,
    { merge: true },
  );
};

export const removeDenial = async (denialId: string): Promise<void> => {
  await deleteDoc(doc(denialsCollection, denialId));
};
