'use client';

import { payerScorecards } from '../../../lib/data';
import { Heatmap } from '../../components/Heatmap';

const customerExperienceHeatmap = {
  rows: ['Pre-auth', 'Claims', 'Appeals'],
  columns: ['Blue Horizon', 'Pioneer Health', 'United Wellness'],
  values: [
    [0.82, 0.9, 0.74],
    [0.77, 0.86, 0.69],
    [0.61, 0.73, 0.58]
  ]
};

const experienceDisplay = customerExperienceHeatmap.values.map((row) =>
  row.map((value) => `${(value * 100).toFixed(0)}%`)
);

export default function PayerScorecardsPage() {
  return (
    <>
      <section className="card">
        <h2>Payer Scorecards</h2>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Payer</th>
                <th>Approval Rate</th>
                <th>Denial Rate</th>
                <th>Avg. Turnaround</th>
                <th>NPS</th>
              </tr>
            </thead>
            <tbody>
              {payerScorecards.map((scorecard) => (
                <tr key={scorecard.payer}>
                  <td>{scorecard.payer}</td>
                  <td>
                    <span className="badge positive">{(scorecard.approvalRate * 100).toFixed(1)}%</span>
                  </td>
                  <td>
                    <span className="badge negative">{(scorecard.denialRate * 100).toFixed(1)}%</span>
                  </td>
                  <td>{scorecard.avgTurnaround} days</td>
                  <td>{scorecard.nps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Experience Heatmap</h2>
        <Heatmap
          rows={customerExperienceHeatmap.rows}
          columns={customerExperienceHeatmap.columns}
          values={customerExperienceHeatmap.values}
          displayValues={experienceDisplay}
        />
      </section>
    </>
  );
}
