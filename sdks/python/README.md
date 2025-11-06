# Python SDK Stub

```python
from sdks.python.appeals_client import AppealsClient

payload = {
    "draft_id": "A-1001",
    "patient_first_name": "Avery",
    "patient_last_name": "Quinn",
    "member_id": "M123",
    "payer_name": "Northwind Payer",
    "provider_npi": "1417999999",
    "denial_code": "D123",
    "scenario": "lack_of_medical_necessity",
    "clinical_summary": "Therapy is supported by three cycles of PT.",
}

with AppealsClient(base_url="http://localhost:8000") as client:
    draft = client.draft(payload)
    submission = client.submit({**payload, "submitted_by": "clinical@denialrx.ai"})
    status = client.status(submission["appeal_id"])
```
