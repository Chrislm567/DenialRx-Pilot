'use client';

import { useMemo, useState } from 'react';
import { claims } from '../../../lib/data';
import { ExportButton } from '../../components/ExportButton';

const stateOptions = ['All', ...Array.from(new Set(claims.map((claim) => claim.state)))];
const payerOptions = ['All', ...Array.from(new Set(claims.map((claim) => claim.payer)))];
const statusOptions = ['All', ...Array.from(new Set(claims.map((claim) => claim.status)))];

export default function ClaimsExplorerPage() {
  const [filters, setFilters] = useState({ state: 'All', payer: 'All', status: 'All' });

  const filteredClaims = useMemo(() => {
    return claims.filter((claim) => {
      if (filters.state !== 'All' && claim.state !== filters.state) return false;
      if (filters.payer !== 'All' && claim.payer !== filters.payer) return false;
      if (filters.status !== 'All' && claim.status !== filters.status) return false;
      return true;
    });
  }, [filters]);

  const updateState = (name: 'state' | 'payer' | 'status', value: string) => {
    setFilters((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <>
      <section className="card">
        <h2>Claims Explorer</h2>
        <div className="filters">
          <label htmlFor="state">
            <span>State</span>
            <select
              id="state"
              name="state"
              value={filters.state}
              onChange={(event) => updateState('state', event.target.value)}
            >
              {stateOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="payer">
            <span>Payer</span>
            <select
              id="payer"
              name="payer"
              value={filters.payer}
              onChange={(event) => updateState('payer', event.target.value)}
            >
              {payerOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="status">
            <span>Status</span>
            <select
              id="status"
              name="status"
              value={filters.status}
              onChange={(event) => updateState('status', event.target.value)}
            >
              {statusOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="metric" aria-live="polite">
          <span className="label">Showing</span>
          <span className="value" data-testid="count">
            {filteredClaims.length}
          </span>
          <span className="label">of {claims.length} claims</span>
        </div>
        <ExportButton filename="claims.csv" data={filteredClaims} />
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Payer</th>
                <th>State</th>
                <th>Specialty</th>
                <th>Status</th>
                <th>Reimbursement</th>
                <th>Turnaround (days)</th>
              </tr>
            </thead>
            <tbody>
              {filteredClaims.map((claim) => (
                <tr key={claim.id}>
                  <td>{claim.id}</td>
                  <td>{claim.payer}</td>
                  <td>{claim.state}</td>
                  <td>{claim.specialty}</td>
                  <td>{claim.status}</td>
                  <td>${claim.reimbursement.toLocaleString()}</td>
                  <td>{claim.turnaroundDays}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
