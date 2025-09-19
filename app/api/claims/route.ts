import { NextResponse } from 'next/server';
import { withApiAuthRequired } from '@auth0/nextjs-auth0';
import { claims } from '../../../lib/data';
import { apiRequestHistogram } from '../../../lib/metrics';
import logger from '../../../lib/logging';

export const GET = withApiAuthRequired(async (request) => {
  const timer = apiRequestHistogram.startTimer({ route: '/api/claims', method: 'GET' });
  try {
    logger.info({ route: '/api/claims', count: claims.length }, 'claims dataset requested');
    return NextResponse.json({ data: claims });
  } catch (error) {
    logger.error({ error }, 'failed to load claims');
    throw error;
  } finally {
    const status = 200;
    timer({ status });
  }
});
