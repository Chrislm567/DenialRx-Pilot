import { useEffect, useState } from 'react';

import { mockClaims } from '../../constants/mockClaims';
import type { Claim } from '../../types/billing';
import { listClaims } from '../services/firestore/claimService';

export type ClaimsDataSource = 'firestore' | 'mock';

interface UseClaimsResult {
  claims: Claim[];
  dataSource: ClaimsDataSource;
  isLoading: boolean;
}

export function useClaims(): UseClaimsResult {
  const [claims, setClaims] = useState<Claim[]>(mockClaims);
  const [dataSource, setDataSource] = useState<ClaimsDataSource>('mock');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const loadClaims = async () => {
      try {
        const firestoreClaims = await listClaims();

        if (isMounted && firestoreClaims.length > 0) {
          setClaims(firestoreClaims);
          setDataSource('firestore');
        }
      } catch (error) {
        console.warn('DenialRx is using mock claims because Firestore is unavailable.', error);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void loadClaims();

    return () => {
      isMounted = false;
    };
  }, []);

  return { claims, dataSource, isLoading };
}
