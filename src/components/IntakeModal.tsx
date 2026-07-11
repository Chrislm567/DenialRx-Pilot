import { useState, type ChangeEvent, type FormEvent } from 'react';

import { denialDatabase } from '../../constants/denialDatabase';
import type { Claim } from '../../types/billing';
import {
  parseRemittanceText,
  type SupportedCarcCode,
} from '../utils/textParser';

export interface IntakeSubmission {
  claim: Omit<Claim, 'workspaceId'>;
  carcCode: SupportedCarcCode;
}

interface IntakeModalProps {
  isOpen: boolean;
  existingClaimIds: string[];
  onClose: () => void;
  onSubmit: (submission: IntakeSubmission) => void;
}

const initialForm = {
  claimId: '',
  mockPatientName: '',
  insurancePayer: '',
  billedAmount: '',
  carcCode: '197' as SupportedCarcCode,
  rawText: '',
};

type IntakeForm = typeof initialForm;

export function IntakeModal({ isOpen, existingClaimIds, onClose, onSubmit }: IntakeModalProps) {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState<IntakeForm>(initialForm);

  if (!isOpen) return null;

  const updateField = (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const name = event.target.name as keyof IntakeForm;
    const value = event.target.value as IntakeForm[typeof name];
    setForm((current) => ({ ...current, [name]: value }));
  };

  const handleRawText = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const rawText = event.target.value;
    const parsed = parseRemittanceText(rawText);
    setForm((current) => ({
      ...current,
      rawText,
      claimId: parsed.claimId ?? current.claimId,
      carcCode: parsed.carcCode ?? current.carcCode,
    }));
  };

  const closeAndReset = () => {
    setStep(1);
    setForm(initialForm);
    onClose();
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (existingClaimIds.includes(form.claimId.trim())) return;

    onSubmit({
      claim: {
        id: form.claimId.trim(),
        mockPatientName: form.mockPatientName.trim(),
        insurancePayer: form.insurancePayer.trim(),
        dateOfService: new Date().toISOString().slice(0, 10),
        totalBilled: Number(form.billedAmount),
        status: 'New',
      },
      carcCode: form.carcCode,
    });
    closeAndReset();
  };

  const duplicateId = existingClaimIds.includes(form.claimId.trim());
  const canContinue = Boolean(
    form.claimId.trim() && form.mockPatientName.trim() && form.insurancePayer.trim(),
  );
  const canSubmit = Number(form.billedAmount) > 0 && !duplicateId;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/40" role="dialog" aria-modal="true">
      <div className="h-full w-full max-w-xl overflow-y-auto bg-white shadow-2xl">
        <form onSubmit={handleSubmit} className="flex min-h-full flex-col">
          <header className="border-b border-slate-200 px-6 py-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-700">Step {step} of 2</p>
                <h2 className="mt-1 text-2xl font-semibold text-slate-950">New denied claim</h2>
                <p className="mt-1 text-sm text-slate-500">Mock intake only. Do not enter real patient information.</p>
              </div>
              <button type="button" onClick={closeAndReset} className="text-sm font-semibold text-slate-500 hover:text-slate-900">Close</button>
            </div>
          </header>

          <div className="flex-1 space-y-5 px-6 py-6">
            {step === 1 ? (
              <>
                <label className="block text-sm font-medium text-slate-700">Claim ID
                  <input name="claimId" value={form.claimId} onChange={updateField} required className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" />
                </label>
                {duplicateId && <p className="text-sm font-medium text-red-600">This claim ID already exists.</p>}
                <label className="block text-sm font-medium text-slate-700">Mock Patient Name
                  <input name="mockPatientName" value={form.mockPatientName} onChange={updateField} required className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" />
                </label>
                <label className="block text-sm font-medium text-slate-700">Insurance Payer
                  <input name="insurancePayer" value={form.insurancePayer} onChange={updateField} required className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" />
                </label>
              </>
            ) : (
              <>
                <label className="block text-sm font-medium text-slate-700">Billed Amount
                  <input name="billedAmount" type="number" min="0.01" step="0.01" value={form.billedAmount} onChange={updateField} required className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" />
                </label>
                <label className="block text-sm font-medium text-slate-700">CARC Denial Code
                  <select name="carcCode" value={form.carcCode} onChange={updateField} className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100">
                    {denialDatabase.map((template) => <option key={template.carcCode} value={template.carcCode}>{template.carcCode} — {template.denialReason}</option>)}
                  </select>
                </label>
                <label className="block text-sm font-medium text-slate-700">Raw Text Remittance Input
                  <textarea value={form.rawText} onChange={handleRawText} rows={7} placeholder="Example: REMIT NO: 20481 ... CO-197" className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5 font-mono text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100" />
                </label>
                <p className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-700">Detected markers auto-fill Claim ID and CARC code.</p>
              </>
            )}
          </div>

          <footer className="flex items-center justify-between border-t border-slate-200 px-6 py-4">
            <button type="button" onClick={() => setStep(1)} disabled={step === 1} className="text-sm font-semibold text-slate-600 disabled:opacity-40">Back</button>
            {step === 1 ? (
              <button type="button" onClick={() => setStep(2)} disabled={!canContinue || duplicateId} className="rounded-lg bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-40">Continue</button>
            ) : (
              <button type="submit" disabled={!canSubmit} className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-40">Add Claim</button>
            )}
          </footer>
        </form>
      </div>
    </div>
  );
}
