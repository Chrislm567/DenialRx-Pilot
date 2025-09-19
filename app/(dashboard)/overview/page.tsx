import { overviewMetrics, alertingBacklog, heatmapData } from '../../../lib/data';
import { Heatmap } from '../../components/Heatmap';

export default function OverviewPage() {
  return (
    <>
      <section className="card">
        <h2>Pulse</h2>
        <div className="metrics-grid">
          {overviewMetrics.map((metric) => (
            <div className="metric" key={metric.label}>
              <span className="label">{metric.label}</span>
              <span className="value">{metric.value}</span>
              <span className="label">{metric.trend}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Denial Density</h2>
        <Heatmap rows={heatmapData.rows} columns={heatmapData.columns} values={heatmapData.values} />
      </section>

      <section className="card">
        <h2>Active Alerts</h2>
        <div className="alert-banner">
          <strong>Tracking {alertingBacklog.length} alert streams</strong>
        </div>
        <div className="metrics-grid" style={{ marginTop: '1.5rem' }}>
          {alertingBacklog.map((alert) => (
            <div className="metric" key={alert.id}>
              <span className="label">{alert.id}</span>
              <span className="value" style={{ fontSize: '1rem' }}>
                {alert.title}
              </span>
              <span className="label">{alert.detail}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
