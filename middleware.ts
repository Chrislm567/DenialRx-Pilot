import { withMiddlewareAuthRequired } from '@auth0/nextjs-auth0/edge';
import { NextResponse } from 'next/server';

const isAuthDisabled = process.env.AUTH_DISABLED === 'true';

const middlewareImpl = isAuthDisabled
  ? async () => NextResponse.next()
  : withMiddlewareAuthRequired(async function middleware() {
      return NextResponse.next();
    });

export default middlewareImpl;

export const config = {
  matcher: ['/((?!api/auth|api/metrics|_next/static|_next/image|favicon.ico).*)']
};
