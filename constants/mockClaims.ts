import type { Claim } from '../types/billing';

export const mockClaims: Claim[] = [
  {
    id: 'CLM-1001',
    mockPatientName: 'Mock Patient Alpha',
    insurancePayer: 'Mock Atlantic Health Plan',
    dateOfService: '2026-05-12',
    totalBilled: 4850,
    status: 'New',
  },
  {
    id: 'CLM-1002',
    mockPatientName: 'Mock Patient Bravo',
    insurancePayer: 'Mock GulfCare Insurance',
    dateOfService: '2026-05-18',
    totalBilled: 1275.5,
    status: 'In Progress',
  },
  {
    id: 'CLM-1003',
    mockPatientName: 'Mock Patient Charlie',
    insurancePayer: 'Mock Sunshine Benefit Network',
    dateOfService: '2026-05-27',
    totalBilled: 9320,
    status: 'Appealed',
  },
  {
    id: 'CLM-1004',
    mockPatientName: 'Mock Patient Delta',
    insurancePayer: 'Mock Coastal Administrators',
    dateOfService: '2026-06-03',
    totalBilled: 2140.75,
    status: 'In Progress',
  },
  {
    id: 'CLM-1005',
    mockPatientName: 'Mock Patient Echo',
    insurancePayer: 'Mock Peninsula Health Plan',
    dateOfService: '2026-06-11',
    totalBilled: 675,
    status: 'Paid',
  },
];
