import { useEffect, useState } from 'react';

import { mockClaims } from '../../constants/mockClaims';
import type { Claim } from '../../types/billing';
import { getCurrentWorkspaceId } from '../services/auth/authService';
import { listClaims } from '../services/firestore/claimService';

export type ClaimsDataSource = 'firestore' | 'mock';

interface UseClaimsResult {
  claims: Claim[];
  dataSource: ClaimsDataSource;
  isLoading: boolean;
  workspaceId: string;
}

export function useClaims(): UseClaimsResult {
  const workspaceId = getCurrentWorkspaceId();
  const [claims, setClaims] = useState<Claim[]>(mockClaims);
  const [dataSource, setDataSource] = useState<ClaimsDataSource>('mock');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const loadClaims = async () => {
      try {
        const firestoreClaims = await listClaims(workspaceId);

        if (isMounted && firestoreClaims.length > 0) {
          setClaims(firestoreClaims);
          setDataSource('firestore');
        }
      } catch {
        if (isMounted) setDataSource('mock');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    void loadClaims();

    return () => {
      isMounted = false;
    };
  }, [workspaceId]);

  return { claims, dataSource, isLoading, workspaceId };
}
