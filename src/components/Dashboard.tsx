import { useEffect, useMemo, useRef, useState } from 'react';

import { denialDatabase } from '../../constants/denialDatabase';
import type { Claim } from '../../types/billing';
import { useClaims } from '../hooks/useClaims';
import { DenialTable, type DenialTableClaim } from './DenialTable';
import { IntakeModal, type IntakeSubmission } from './IntakeModal';
import { MetricsSummary } from './MetricsSummary';

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

export function Dashboard() {
  const { claims: loadedClaims, dataSource, isLoading } = useClaims();
  const [claims, setClaims] = useState<Claim[]>(loadedClaims);
  const [carcOverrides, setCarcOverrides] = useState<Record<string, string>>({});
  const [isIntakeOpen, setIsIntakeOpen] = useState(false);
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
    hasLocalEdits.current = true;
    setClaims((current) => [...current, claim]);
    setCarcOverrides((current) => ({ ...current, [claim.id]: carcCode }));
  };

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-6 text-slate-950 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-screen-2xl flex-col gap-6">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-blue-700">
              Revenue Recovery Command Center
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight text-slate-950">
              DenialRx Dashboard
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Prioritize active denials, track trapped revenue, and move claims toward appeal.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-500 shadow-sm">
              {isLoading ? 'Checking Firestore…' : sourceLabel}
            </div>
            <button
              type="button"
              onClick={() => setIsIntakeOpen(true)}
              className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Add Denied Claim
            </button>
          </div>
        </header>

        <MetricsSummary claims={claims} />

        <section aria-labelledby="active-denials-heading" className="space-y-3">
          <div>
            <h2 id="active-denials-heading" className="text-lg font-semibold text-slate-950">
              Active Denial Work Queue
            </h2>
            <p className="text-sm text-slate-500">
              {activeClaims.length} claims currently require revenue-cycle action.
            </p>
          </div>
          <DenialTable claims={activeClaims} />
        </section>
      </div>

      <IntakeModal
        isOpen={isIntakeOpen}
        existingClaimIds={claims.map((claim) => claim.id)}
        onClose={() => setIsIntakeOpen(false)}
        onSubmit={handleIntakeSubmit}
      />
    </main>
  );
}
