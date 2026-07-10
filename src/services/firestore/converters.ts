import type {
  DocumentData,
  FirestoreDataConverter,
  QueryDocumentSnapshot,
  SnapshotOptions,
  WithFieldValue,
} from 'firebase/firestore';

import type { AppealTemplate, Claim, Denial } from '../../../types/billing';

const createConverter = <T extends { id: string }>(): FirestoreDataConverter<T> => ({
  toFirestore(modelObject: WithFieldValue<T>): DocumentData {
    const documentData: DocumentData = { ...modelObject };
    delete documentData.id;
    return documentData;
  },
  fromFirestore(
    snapshot: QueryDocumentSnapshot<DocumentData>,
    options: SnapshotOptions,
  ): T {
    return {
      id: snapshot.id,
      ...snapshot.data(options),
    } as T;
  },
});

export const claimConverter = createConverter<Claim>();
export const denialConverter = createConverter<Denial>();

export const appealTemplateConverter: FirestoreDataConverter<AppealTemplate> = {
  toFirestore(modelObject: WithFieldValue<AppealTemplate>): DocumentData {
    return modelObject;
  },
  fromFirestore(
    snapshot: QueryDocumentSnapshot<DocumentData>,
    options: SnapshotOptions,
  ): AppealTemplate {
    return snapshot.data(options) as AppealTemplate;
  },
};
