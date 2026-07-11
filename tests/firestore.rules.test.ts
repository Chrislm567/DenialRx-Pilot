import { readFileSync } from 'node:fs';

import {
  assertFails,
  assertSucceeds,
  initializeTestEnvironment,
  type RulesTestEnvironment,
} from '@firebase/rules-unit-testing';
import { collection, doc, getDoc, getDocs, query, setDoc, where } from 'firebase/firestore';
import { afterAll, beforeAll, beforeEach, describe, it } from 'vitest';

const PROJECT_ID = 'denialrx-rules-test';
let testEnvironment: RulesTestEnvironment;

const mockClaim = (workspaceId: string) => ({
  workspaceId,
  mockPatientName: 'Mock Patient Rules',
  insurancePayer: 'Mock Rules Payer',
  dateOfService: '2026-07-11',
  totalBilled: 1200,
  status: 'New',
});

const seedMembership = async (
  workspaceId: string,
  userId: string,
  role: 'admin' | 'biller' | 'viewer',
) => {
  await testEnvironment.withSecurityRulesDisabled(async (context) => {
    await setDoc(doc(context.firestore(), 'workspaces', workspaceId, 'members', userId), {
      workspaceId,
      userId,
      role,
      createdAt: '2026-07-11T00:00:00.000Z',
    });
  });
};

beforeAll(async () => {
  testEnvironment = await initializeTestEnvironment({
    projectId: PROJECT_ID,
    firestore: { rules: readFileSync('firestore.rules', 'utf8') },
  });
});

beforeEach(async () => testEnvironment.clearFirestore());
afterAll(async () => testEnvironment.cleanup());

describe('Firestore workspace roles', () => {
  it('allows personal workspace admin bootstrap', async () => {
    const db = testEnvironment.authenticatedContext('user-a').firestore();
    const memberRef = doc(db, 'workspaces', 'user-a', 'members', 'user-a');

    await assertSucceeds(setDoc(memberRef, {
      workspaceId: 'user-a', userId: 'user-a', role: 'admin', createdAt: '2026-07-11',
    }));
  });

  it('allows billers to create and viewers to read workspace claims', async () => {
    await seedMembership('clinic-a', 'biller-a', 'biller');
    await seedMembership('clinic-a', 'viewer-a', 'viewer');
    const billerDb = testEnvironment.authenticatedContext('biller-a').firestore();
    const claimRef = doc(billerDb, 'claims', 'CLM-A');

    await assertSucceeds(setDoc(claimRef, mockClaim('clinic-a')));
    const viewerDb = testEnvironment.authenticatedContext('viewer-a').firestore();
    await assertSucceeds(getDoc(doc(viewerDb, 'claims', 'CLM-A')));
    await assertFails(setDoc(doc(viewerDb, 'claims', 'CLM-B'), mockClaim('clinic-a')));
  });

  it('rejects cross-workspace reads and foreign ownership writes', async () => {
    await seedMembership('clinic-a', 'user-a', 'admin');
    await testEnvironment.withSecurityRulesDisabled(async (context) => {
      await setDoc(doc(context.firestore(), 'claims', 'CLM-B'), mockClaim('clinic-b'));
    });
    const db = testEnvironment.authenticatedContext('user-a').firestore();

    await assertFails(getDoc(doc(db, 'claims', 'CLM-B')));
    await assertFails(setDoc(doc(db, 'claims', 'CLM-C'), mockClaim('clinic-b')));
  });

  it('allows scoped queries and rejects unscoped collection queries', async () => {
    await seedMembership('clinic-a', 'user-a', 'viewer');
    await testEnvironment.withSecurityRulesDisabled(async (context) => {
      await setDoc(doc(context.firestore(), 'claims', 'CLM-A'), mockClaim('clinic-a'));
      await setDoc(doc(context.firestore(), 'claims', 'CLM-B'), mockClaim('clinic-b'));
    });
    const db = testEnvironment.authenticatedContext('user-a').firestore();
    const scopedQuery = query(collection(db, 'claims'), where('workspaceId', '==', 'clinic-a'));

    await assertSucceeds(getDocs(scopedQuery));
    await assertFails(getDocs(collection(db, 'claims')));
  });
});
