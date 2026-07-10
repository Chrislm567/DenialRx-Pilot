import { useEffect, useMemo, useState, type ChangeEvent } from 'react';

import type { DenialTableClaim } from './DenialTable';
import { renderAppealLetter } from '../utils/appealLetter';

interface ClaimDetailsPaneProps {
  claim: DenialTableClaim | null;
  onClose: () => void;
}

export function ClaimDetailsPane({ claim, onClose }: ClaimDetailsPaneProps) {
  const [draftClaim, setDraftClaim] = useState(claim);
  const generated = useMemo(
    () => (draftClaim ? renderAppealLetter(draftClaim, draftClaim.denialCarc) : null),
    [draftClaim],
  );
  const [letterText, setLetterText] = useState('');

  useEffect(() => {
    setDraftClaim(claim);
  }, [claim]);

  useEffect(() => {
    setLetterText(generated?.letterText ?? '');
  }, [generated]);

  if (!claim || !draftClaim || !generated) return null;

  const updateField = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setDraftClaim((current) =>
      current
        ? {
            ...current,
            [name]: name === 'totalBilled' ? Number(value) : value,
          }
        : current,
    );
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/40" role="dialog" aria-modal="true">
      <section className="ml-auto flex h-full w-full max-w-6xl flex-col bg-white shadow-2xl">
        <header className="flex items-start justify-between border-b border-slate-200 px-6 py-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-700">
              Appeal Workbench
            </p>
            <h2 className="mt-1 text-2xl font-semibold text-slate-950">Claim {claim.id}</h2>
            <p className="mt-1 text-sm text-slate-500">CARC {claim.denialCarc} mapped to a live appeal template.</p>
          </div>
          <button type="button" onClick={onClose} className="text-sm font-semibold text-slate-500 hover:text-slate-900">
            Close
          </button>
        </header>

        <div className="grid flex-1 overflow-hidden lg:grid-cols-[0.9fr_1.4fr]">
          <aside className="overflow-y-auto border-b border-slate-200 bg-slate-50 p-6 lg:border-b-0 lg:border-r">
            <div className="space-y-5">
              <label className="block text-sm font-medium text-slate-700">Claim ID
                <input name="id" value={draftClaim.id} onChange={updateField} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5" />
              </label>
              <label className="block text-sm font-medium text-slate-700">Insurance Payer
                <input name="insurancePayer" value={draftClaim.insurancePayer} onChange={updateField} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5" />
              </label>
              <label className="block text-sm font-medium text-slate-700">Total Balance
                <input name="totalBilled" type="number" min="0" step="0.01" value={draftClaim.totalBilled} onChange={updateField} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5" />
              </label>
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Denial reason</p>
                <p className="mt-2 text-sm font-medium text-slate-900">{generated.template?.denialReason}</p>
              </div>
              <div className="rounded-xl border border-blue-100 bg-blue-50 p-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-blue-700">Argument strategy</p>
                <p className="mt-2 text-sm leading-6 text-blue-900">{generated.template?.argumentStrategy}</p>
              </div>
            </div>
          </aside>

          <div className="flex min-h-0 flex-col p-6">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-950">Live appeal letter</h3>
                <p className="text-sm text-slate-500">Edit the generated argument before export.</p>
              </div>
              <button type="button" onClick={() => setLetterText(generated.letterText)} className="text-sm font-semibold text-blue-700 hover:text-blue-900">
                Reset template
              </button>
            </div>
            <textarea
              value={letterText}
              onChange={(event) => setLetterText(event.target.value)}
              className="min-h-[28rem] flex-1 resize-none rounded-xl border border-slate-300 bg-white p-5 font-serif text-base leading-7 text-slate-900 shadow-inner outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              aria-label="Editable appeal letter"
            />
          </div>
        </div>
      </section>
    </div>
  );
}