import { diag, DiagConsoleLogger, DiagLogLevel } from '@opentelemetry/api';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { BatchSpanProcessor, ConsoleSpanExporter } from '@opentelemetry/sdk-trace-base';

let provider: NodeTracerProvider | undefined;

export function registerOTel() {
  if (typeof window !== 'undefined' || typeof navigator !== 'undefined') {
    return;
  }

  if (provider) {
    return;
  }

  diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.ERROR);

  provider = new NodeTracerProvider({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: 'phase-runner-web',
      environment: process.env.OTEL_ENVIRONMENT ?? 'local'
    })
  });

  const endpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT;

  if (endpoint) {
    import('@opentelemetry/exporter-trace-otlp-http').then(({ OTLPTraceExporter }) => {
      const exporter = new OTLPTraceExporter({ url: endpoint });
      provider?.addSpanProcessor(new BatchSpanProcessor(exporter));
    });
  } else {
    provider.addSpanProcessor(new BatchSpanProcessor(new ConsoleSpanExporter()));
  }

  provider.register();

  process.on('SIGTERM', () => {
    provider
      ?.shutdown()
      .then(() => console.info('[otel] graceful shutdown'))
      .catch((error) => console.error('[otel] shutdown error', error));
  });
}
