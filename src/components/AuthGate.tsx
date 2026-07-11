import type { ReactNode } from 'react';

import { useAuthSession } from '../hooks/useAuthSession';

interface AuthGateProps {
  children: ReactNode;
}

export function AuthGate({ children }: AuthGateProps) {
  const { user, workspaceAccess, isLoading, errorMessage } = useAuthSession();

  if (isLoading) {
    return (
      <main className="grid min-h-screen place-items-center bg-slate-100 px-6">
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-5 text-center shadow-sm">
          <p className="text-sm font-semibold text-slate-900">Starting secure DenialRx workspace…</p>
          <p className="mt-1 text-xs text-slate-500">No patient data is being transmitted.</p>
        </div>
      </main>
    );
  }

  if (errorMessage || !user || !workspaceAccess) {
    return (
      <main className="grid min-h-screen place-items-center bg-slate-100 px-6">
        <div className="max-w-md rounded-xl border border-red-200 bg-white p-6 shadow-sm">
          <h1 className="text-lg font-semibold text-slate-950">Workspace access unavailable</h1>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            {errorMessage ?? 'A secure workspace membership could not be established.'}
          </p>
          <p className="mt-3 text-xs text-slate-500">
            Enable Anonymous Authentication and deploy the current Firestore rules before using this MVP.
          </p>
        </div>
      </main>
    );
  }

  return children;
}
