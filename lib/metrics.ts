import { Counter, Gauge, Histogram, Registry, collectDefaultMetrics } from 'prom-client';

const register = new Registry();
collectDefaultMetrics({ register });

export const apiRequestHistogram = new Histogram({
  name: 'phase_runner_api_request_duration_seconds',
  help: 'Duration histogram of API requests',
  labelNames: ['route', 'method', 'status'],
  buckets: [0.05, 0.1, 0.25, 0.5, 1, 2, 5]
});

export const etlThroughputGauge = new Gauge({
  name: 'phase_runner_etl_throughput_records',
  help: 'ETL pipeline throughput',
  labelNames: ['pipeline']
});

export const errorBudgetBurnCounter = new Counter({
  name: 'phase_runner_error_budget_burn_total',
  help: 'Error budget burn events',
  labelNames: ['service']
});

register.registerMetric(apiRequestHistogram);
register.registerMetric(etlThroughputGauge);
register.registerMetric(errorBudgetBurnCounter);

export function getMetricsRegistry() {
  return register;
}
