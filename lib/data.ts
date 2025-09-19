export type ClaimRecord = {
  id: string;
  payer: string;
  state: string;
  specialty: string;
  submissionDate: string;
  status: 'Approved' | 'Denied' | 'Pending';
  denialReason?: string;
  reimbursement: number;
  turnaroundDays: number;
};

export type Scorecard = {
  payer: string;
  approvalRate: number;
  denialRate: number;
  avgTurnaround: number;
  nps: number;
};

export const claims: ClaimRecord[] = [
  {
    id: 'CLM-10001',
    payer: 'Blue Horizon',
    state: 'CA',
    specialty: 'Cardiology',
    submissionDate: '2024-03-01',
    status: 'Approved',
    reimbursement: 1520,
    turnaroundDays: 5
  },
  {
    id: 'CLM-10002',
    payer: 'Pioneer Health',
    state: 'WA',
    specialty: 'Endocrinology',
    submissionDate: '2024-03-02',
    status: 'Denied',
    denialReason: 'Missing chart notes',
    reimbursement: 0,
    turnaroundDays: 8
  },
  {
    id: 'CLM-10003',
    payer: 'United Wellness',
    state: 'TX',
    specialty: 'Oncology',
    submissionDate: '2024-03-04',
    status: 'Pending',
    reimbursement: 0,
    turnaroundDays: 10
  },
  {
    id: 'CLM-10004',
    payer: 'Blue Horizon',
    state: 'CA',
    specialty: 'Orthopedics',
    submissionDate: '2024-03-04',
    status: 'Denied',
    denialReason: 'Eligibility expired',
    reimbursement: 0,
    turnaroundDays: 11
  },
  {
    id: 'CLM-10005',
    payer: 'Pioneer Health',
    state: 'AZ',
    specialty: 'Neurology',
    submissionDate: '2024-03-05',
    status: 'Approved',
    reimbursement: 1875,
    turnaroundDays: 6
  }
];

export const payerScorecards: Scorecard[] = [
  {
    payer: 'Blue Horizon',
    approvalRate: 0.78,
    denialRate: 0.14,
    avgTurnaround: 8,
    nps: 32
  },
  {
    payer: 'Pioneer Health',
    approvalRate: 0.83,
    denialRate: 0.1,
    avgTurnaround: 6,
    nps: 54
  },
  {
    payer: 'United Wellness',
    approvalRate: 0.71,
    denialRate: 0.18,
    avgTurnaround: 9,
    nps: 21
  }
];

export const heatmapData = {
  rows: ['Cardiology', 'Endocrinology', 'Oncology', 'Orthopedics', 'Neurology'],
  columns: ['Blue Horizon', 'Pioneer Health', 'United Wellness'],
  values: [
    [0.9, 0.81, 0.74],
    [0.72, 0.69, 0.66],
    [0.68, 0.61, 0.58],
    [0.61, 0.57, 0.49],
    [0.94, 0.87, 0.8]
  ]
};

export const overviewMetrics = [
  {
    label: 'Current Approval Rate',
    value: '82.4%',
    trend: '+4.1% WoW'
  },
  {
    label: 'Denied Claims',
    value: '124',
    trend: '-12.7% MoM'
  },
  {
    label: 'Avg. Turnaround (days)',
    value: '7.2',
    trend: '-0.6 days'
  },
  {
    label: 'Revenue at Risk',
    value: '$284K',
    trend: '-$22K WoW'
  }
];

export const alertingBacklog = [
  {
    id: 'ALRT-404',
    title: 'Eligibility denials breached SLO',
    detail: 'Cardiology eligibility denials trending 8% above threshold for 48h.'
  },
  {
    id: 'ALRT-417',
    title: 'RPA bot error budget',
    detail: 'Automation failure budget consumed at 67% for the week.'
  }
];
