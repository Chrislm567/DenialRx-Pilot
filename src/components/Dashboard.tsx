import { useEffect, useMemo, useRef, useState } from 'react';

import { denialDatabase } from '../../constants/denialDatabase';
import type { Claim } from '../../types/billing';
import { useClaims } from '../hooks/useClaims';
import { updateClaim } from '../services/firestore/claimService';
import { ClaimDetailsPane } from './ClaimDetailsPane';
import { DenialTable, type DenialTableClaim } from './DenialTable';
import { IntakeModal, type IntakeSubmission } from './IntakeModal';
import { MetricsSummary } from './MetricsSummary';
import {
  NotificationBanner,
  type NotificationTone,
} from './NotificationBanner';

const buildActiveClaims = (
  claims: Claim[],
  carcOverrides: Record<string, string>,
): DenialTableClaim[] =>
  claims
    .filter((claim) => claim.status !== 'Paid')
    .map((claim, index) => ({
      ...claim,
      denialCarc:
        carcOverrides[claim.id] ??
        denialDatabase[index % denialDatabase.length]?.carcCode ??
        '—',
    }));

interface DashboardNotice {
  message: string;
  tone: NotificationTone;
}

export function Dashboard() {
  const { claims: loadedClaims, dataSource, isLoading, workspaceId } = useClaims();
  const [claims, setClaims] = useState<Claim[]>(loadedClaims);
  const [carcOverrides, setCarcOverrides] = useState<Record<string, string>>({});
  const [isIntakeOpen, setIsIntakeOpen] = useState(false);
  const [selectedClaim, setSelectedClaim] = useState<DenialTableClaim | null>(null);
  const [notice, setNotice] = useState<DashboardNotice | null>(null);
  const hasLocalEdits = useRef(false);

  useEffect(() => {
    if (!hasLocalEdits.current) setClaims(loadedClaims);
  }, [loadedClaims]);

  const activeClaims = useMemo(
    () => buildActiveClaims(claims, carcOverrides),
    [claims, carcOverrides],
  );
  const sourceLabel = dataSource === 'firestore' ? 'Firestore data' : 'Mock operational data';

  const handleIntakeSubmit = ({ claim, carcCode }: IntakeSubmission) => {
    const workspaceClaim: Claim = { ...claim, workspaceId };
    hasLocalEdits.current = true;
    setClaims((current) => [...current, workspaceClaim]);
    setCarcOverrides((current) => ({ ...current, [workspaceClaim.id]: carcCode }));
    setNotice({ message: `Claim ${workspaceClaim.id} added to the local work queue.`, tone: 'success' });
  };

  const handleAppealExport = async (claimId: string) => {
    hasLocalEdits.current = true;
    setClaims((current) =>
      current.map((claim) =>
        claim.id === claimId ? { ...claim, status: 'Appealed' } : claim,
      ),
    );
    setSelectedClaim((current) =>
      current?.id === claimId ? { ...current, status: 'Appealed' } : current,
    );

    if (dataSource !== 'firestore') {
      setNotice({ message: `Appeal exported. Claim ${claimId} is marked Appealed locally.`, tone: 'success' });
      return;
    }

    try {
      await updateClaim(claimId, workspaceId, { status: 'Appealed' });
      setNotice({ message: `Appeal exported and claim ${claimId} was saved to Firestore.`, tone: 'success' });
    } catch {
      setNotice({ message: `Appeal exported, but Firestore could not save claim ${claimId}. The local status is still Appealed.`, tone: 'warning' });
    }
  };

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-6 text-slate-950 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-screen-2xl flex-col gap-6">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">Revenue Recovery Command Center</p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight text-slate-950">DenialRx Dashboard</h1>
            <p className="mt-1 text-sm text-slate-600">Prioritize active denials, track trapped revenue, and move claims toward appeal.</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-500 shadow-sm">{isLoading ? 'Checking Firestore…' : sourceLabel}</div>
            <button type="button" onClick={() => setIsIntakeOpen(true)} className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">Add Denied Claim</button>
          </div>
        </header>

        {notice && <NotificationBanner {...notice} onDismiss={() => setNotice(null)} />}
        <MetricsSummary claims={claims} />

        <section aria-labelledby="active-denials-heading" className="space-y-3">
          <div>
            <h2 id="active-denials-heading" className="text-lg font-semibold text-slate-950">Active Denial Work Queue</h2>
            <p className="text-sm text-slate-500">{activeClaims.length} claims currently require revenue-cycle action.</p>
          </div>
          <DenialTable claims={activeClaims} onSelectClaim={setSelectedClaim} />
        </section>
      </div>

      <IntakeModal isOpen={isIntakeOpen} existingClaimIds={claims.map((claim) => claim.id)} onClose={() => setIsIntakeOpen(false)} onSubmit={handleIntakeSubmit} />
      <ClaimDetailsPane claim={selectedClaim} onClose={() => setSelectedClaim(null)} onExportSuccess={handleAppealExport} />
    </main>
  );
}
