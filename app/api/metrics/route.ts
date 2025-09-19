import { NextResponse } from 'next/server';
import { getMetricsRegistry } from '../../../lib/metrics';

export async function GET() {
  const register = getMetricsRegistry();
  const body = await register.metrics();
  return new NextResponse(body, {
    status: 200,
    headers: {
      'Content-Type': register.contentType
    }
  });
}
