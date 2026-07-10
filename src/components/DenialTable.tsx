import type { Claim, ClaimStatus } from '../../types/billing';

export interface DenialTableClaim extends Claim {
  denialCarc: string;
}

interface DenialTableProps {
  claims: DenialTableClaim[];
  onSelectClaim: (claim: DenialTableClaim) => void;
}

const statusClasses: Record<Exclude<ClaimStatus, 'Paid'>, string> = {
  New: 'border-red-200 bg-red-50 text-red-700',
  'In Progress': 'border-amber-200 bg-amber-50 text-amber-700',
  Appealed: 'border-blue-200 bg-blue-50 text-blue-700',
};

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
});

export function DenialTable({ claims, onSelectClaim }: DenialTableProps) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-left">
          <thead className="bg-slate-50">
            <tr>
              {['Claim ID', 'Mock Patient', 'Payer', 'Denial CARC', 'Status', 'Total Balance', 'Actions'].map(
                (heading) => (
                  <th
                    key={heading}
                    scope="col"
                    className="whitespace-nowrap px-5 py-3 text-xs font-semibold uppercase tracking-wider text-slate-500"
                  >
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {claims.map((claim) => (
              <tr key={claim.id} className="transition hover:bg-slate-50/80">
                <td className="whitespace-nowrap px-5 py-4 text-sm font-semibold text-slate-900">
                  {claim.id}
                </td>
                <td className="whitespace-nowrap px-5 py-4 text-sm text-slate-700">
                  {claim.mockPatientName}
                </td>
                <td className="min-w-56 px-5 py-4 text-sm text-slate-700">
                  {claim.insurancePayer}
                </td>
                <td className="whitespace-nowrap px-5 py-4 text-sm font-medium text-slate-900">
                  CARC {claim.denialCarc}
                </td>
                <td className="whitespace-nowrap px-5 py-4">
                  <span
                    className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${statusClasses[claim.status as Exclude<ClaimStatus, 'Paid'>]}`}
                  >
                    {claim.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-5 py-4 text-sm font-semibold tabular-nums text-slate-900">
                  {currencyFormatter.format(claim.totalBilled)}
                </td>
                <td className="whitespace-nowrap px-5 py-4">
                  <button
                    type="button"
                    onClick={() => onSelectClaim(claim)}
                    className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}