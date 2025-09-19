import { handleAuth } from '@auth0/nextjs-auth0';
import type { NextApiRequest, NextApiResponse } from 'next';

const isAuthDisabled = process.env.AUTH_DISABLED === 'true';

const disabledAuthHandler = (req: NextApiRequest, res: NextApiResponse) => {
  const actionParam = req.query.auth0;
  const action = Array.isArray(actionParam) ? actionParam[0] : actionParam ?? '';

  switch (action) {
    case 'me':
      res.status(200).json({
        user: {
          sub: 'auth0|local-dev',
          name: 'Local Developer',
          email: 'local.dev@example.com',
        }
      });
      return;
    case 'login':
    case 'logout':
    case 'callback':
      res.writeHead(302, { Location: '/' });
      res.end();
      return;
    default:
      res.status(404).json({ error: 'Not found' });
  }
};

const handler = isAuthDisabled ? disabledAuthHandler : handleAuth();

export default handler;
