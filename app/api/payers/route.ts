import { NextResponse } from 'next/server';
import { withApiAuthRequired } from '@auth0/nextjs-auth0';
import { payerScorecards } from '../../../lib/data';
import { apiRequestHistogram } from '../../../lib/metrics';
import logger from '../../../lib/logging';

export const GET = withApiAuthRequired(async () => {
  const timer = apiRequestHistogram.startTimer({ route: '/api/payers', method: 'GET' });
  try {
    logger.info({ route: '/api/payers', count: payerScorecards.length }, 'payer scorecards requested');
    return NextResponse.json({ data: payerScorecards });
  } catch (error) {
    logger.error({ error }, 'failed to load payers');
    throw error;
  } finally {
    timer({ status: 200 });
  }
});
