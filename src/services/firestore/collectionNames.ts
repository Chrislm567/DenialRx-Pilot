export const FIRESTORE_COLLECTIONS = {
  claims: 'claims',
  denials: 'denials',
  appealTemplates: 'appealTemplates',
} as const;

export type FirestoreCollectionName =
  (typeof FIRESTORE_COLLECTIONS)[keyof typeof FIRESTORE_COLLECTIONS];
