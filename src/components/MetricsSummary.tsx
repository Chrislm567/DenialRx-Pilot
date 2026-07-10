import type { Claim } from '../../types/billing';

interface MetricsSummaryProps {
  claims: Claim[];
}

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

export function MetricsSummary({ claims }: MetricsSummaryProps) {
  const activeClaims = claims.filter((claim) => claim.status !== 'Paid');
  const trappedDollars = activeClaims.reduce(
    (total, claim) => total + claim.totalBilled,
    0,
  );
  const appealedCount = claims.filter((claim) => claim.status === 'Appealed').length;

  const metrics = [
    {
      label: 'Dollars Trapped in Denials',
      value: currencyFormatter.format(trappedDollars),
    },
    {
      label: 'Active Claims',
      value: activeClaims.length.toLocaleString('en-US'),
    },
    {
      label: 'Moved to Appealed',
      value: appealedCount.toLocaleString('en-US'),
    },
  ];

  return (
    <section
      aria-label="Executive denial metrics"
      className="grid overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm md:grid-cols-3"
    >
      {metrics.map((metric, index) => (
        <div
          key={metric.label}
          className={`px-6 py-5 ${index > 0 ? 'border-t border-slate-200 md:border-l md:border-t-0' : ''}`}
        >
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
            {metric.label}
          </p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
            {metric.value}
          </p>
        </div>
      ))}
    </section>
  );
}
