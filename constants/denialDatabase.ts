import type { AppealTemplate } from '../types/billing';

export const denialDatabase: AppealTemplate[] = [
  {
    carcCode: '197',
    denialReason: 'Pre-certification or prior authorization was missing.',
    argumentStrategy:
      'Prove an emergent or routing exception, or attach retrospective medical documentation supporting authorization.',
    standardLetterBody:
      'Re: Claim {{claimId}} | Balance {{balance}}\n\nTo {{payer}} Appeals Department:\n\nWe request reconsideration of the denial assigned under CARC 197. The submitted service met the applicable emergent, routing, or retrospective review exception criteria. Supporting documentation is attached to establish the clinical circumstances, service timeline, and medical basis for retrospective authorization. Please reprocess claim {{claimId}} for the disputed balance of {{balance}} after reviewing the enclosed records and authorization evidence.',
  },
  {
    carcCode: '50',
    denialReason: 'The service was not deemed medically necessary.',
    argumentStrategy:
      'Reference objective clinical criteria, payer policy requirements, and recognized treatment guidelines.',
    standardLetterBody:
      'Re: Claim {{claimId}} | Balance {{balance}}\n\nTo {{payer}} Appeals Department:\n\nWe request reconsideration of the denial assigned under CARC 50. The service was medically necessary based on the documented symptoms, objective findings, prior treatment history, and applicable clinical criteria. The attached records demonstrate alignment with recognized guidelines and the payer\'s published coverage requirements. Please review the supporting evidence and reprocess claim {{claimId}} for {{balance}}.',
  },
  {
    carcCode: '29',
    denialReason: 'The timely filing limit expired.',
    argumentStrategy:
      'Present electronic clearinghouse submission records and 277 claim acceptance timeline evidence.',
    standardLetterBody:
      'Re: Claim {{claimId}} | Balance {{balance}}\n\nTo {{payer}} Appeals Department:\n\nWe request reconsideration of the denial assigned under CARC 29. Electronic submission records show that the claim entered the clearinghouse workflow within the applicable filing period. The attached transmission history and 277 claim acknowledgment timeline document timely receipt and acceptance activity. Please apply the timely filing exception and reprocess claim {{claimId}} for {{balance}}.',
  },
];