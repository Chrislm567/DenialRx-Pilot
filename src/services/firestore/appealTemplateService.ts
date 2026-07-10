import {
  collection,
  doc,
  getDoc,
  getDocs,
  setDoc,
} from 'firebase/firestore';

import type { AppealTemplate } from '../../../types/billing';
import { firestoreDb } from '../../lib/firebase';
import { FIRESTORE_COLLECTIONS } from './collectionNames';
import { appealTemplateConverter } from './converters';

const appealTemplatesCollection = collection(
  firestoreDb,
  FIRESTORE_COLLECTIONS.appealTemplates,
).withConverter(appealTemplateConverter);

export const listAppealTemplates = async (): Promise<AppealTemplate[]> => {
  const snapshot = await getDocs(appealTemplatesCollection);
  return snapshot.docs.map((templateDocument) => templateDocument.data());
};

export const getAppealTemplateByCarcCode = async (
  carcCode: string,
): Promise<AppealTemplate | null> => {
  const snapshot = await getDoc(doc(appealTemplatesCollection, carcCode));
  return snapshot.exists() ? snapshot.data() : null;
};

export const saveAppealTemplate = async (
  template: AppealTemplate,
): Promise<void> => {
  await setDoc(doc(appealTemplatesCollection, template.carcCode), template);
};
