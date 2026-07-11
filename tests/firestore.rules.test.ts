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

beforeAll(async () => {
  testEnvironment = await initializeTestEnvironment({
    projectId: PROJECT_ID,
    firestore: { rules: readFileSync('firestore.rules', 'utf8') },
  });
});

beforeEach(async () => {
  await testEnvironment.clearFirestore();
});

afterAll(async () => {
  await testEnvironment.cleanup();
});

describe('Firestore workspace isolation', () => {
  it('allows a user to create and read a claim in their own workspace', async () => {
    const db = testEnvironment.authenticatedContext('workspace-a').firestore();
    const claimRef = doc(db, 'claims', 'CLM-A');

    await assertSucceeds(setDoc(claimRef, mockClaim('workspace-a')));
    await assertSucceeds(getDoc(claimRef));
  });

  it('rejects writes assigned to another workspace', async () => {
    const db = testEnvironment.authenticatedContext('workspace-a').firestore();

    await assertFails(setDoc(doc(db, 'claims', 'CLM-B'), mockClaim('workspace-b')));
  });

  it('rejects cross-workspace document reads', async () => {
    await testEnvironment.withSecurityRulesDisabled(async (context) => {
      await setDoc(doc(context.firestore(), 'claims', 'CLM-B'), mockClaim('workspace-b'));
    });

    const db = testEnvironment.authenticatedContext('workspace-a').firestore();
    await assertFails(getDoc(doc(db, 'claims', 'CLM-B')));
  });

  it('allows workspace-filtered queries and rejects unscoped claim queries', async () => {
    await testEnvironment.withSecurityRulesDisabled(async (context) => {
      await setDoc(doc(context.firestore(), 'claims', 'CLM-A'), mockClaim('workspace-a'));
      await setDoc(doc(context.firestore(), 'claims', 'CLM-B'), mockClaim('workspace-b'));
    });

    const db = testEnvironment.authenticatedContext('workspace-a').firestore();
    const scopedQuery = query(
      collection(db, 'claims'),
      where('workspaceId', '==', 'workspace-a'),
    );

    await assertSucceeds(getDocs(scopedQuery));
    await assertFails(getDocs(collection(db, 'claims')));
  });
});
