import { useMemo, useState } from 'react';

import { denialDatabase } from '../../constants/denialDatabase';
import { mockClaims } from '../../constants/mockClaims';
import type { Claim } from '../../types/billing';
import { DenialTable, type DenialTableClaim } from './DenialTable';
import { MetricsSummary } from './MetricsSummary';

const buildActiveClaims = (claims: Claim[]): DenialTableClaim[] =>
  claims
    .filter((claim) => claim.status !== 'Paid')
    .map((claim, index) => ({
      ...claim,
      denialCarc: denialDatabase[index % denialDatabase.length]?.carcCode ?? '—',
    }));

export function Dashboard() {
  const [claims] = useState<Claim[]>(mockClaims);
  const activeClaims = useMemo(() => buildActiveClaims(claims), [claims]);

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-6 text-slate-950 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-screen-2xl flex-col gap-6">
        <header className="flex flex-col gap-2 border-b border-slate-200 pb-5 sm:flex-row sm:items-end sm:justify-between">
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
          <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-500 shadow-sm">
            Mock operational data only
          </div>
        </header>

        <MetricsSummary claims={claims} />

        <section aria-labelledby="active-denials-heading" className="space-y-3">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 id="active-denials-heading" className="text-lg font-semibold text-slate-950">
                Active Denial Work Queue
              </h2>
              <p className="text-sm text-slate-500">
                {activeClaims.length} claims currently require revenue-cycle action.
              </p>
            </div>
          </div>
          <DenialTable claims={activeClaims} />
        </section>
      </div>
    </main>
  );
}
