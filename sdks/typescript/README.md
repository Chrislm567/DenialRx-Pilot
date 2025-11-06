# TypeScript SDK Stub

```ts
import { AppealsClient } from "./appealsClient";

const client = new AppealsClient("http://localhost:8000");
const draftPayload = {
  draft_id: "TS-401",
  patient_first_name: "Sky",
  patient_last_name: "Rivera",
  member_id: "MEM-99",
  payer_name: "Aster Payer",
  provider_npi: "1417222222",
  denial_code: "PRIOR-404",
  scenario: "prior_auth_missing",
  clinical_summary: "Prior authorization was granted on 2024-03-10.",
};

async function run() {
  const draft = await client.draft(draftPayload);
  const submission = await client.submit({ ...draftPayload, submitted_by: "ops@denialrx.ai" });
  const status = await client.status(submission.appeal_id);
  console.log(status);
}

run();
```
