import { registerOTel } from './otel';

export function register() {
  if (process.env.NEXT_RUNTIME === 'edge') {
    return;
  }
  registerOTel();
}
