import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

N_CLAIMS = 240

payers = ["Aetna", "BlueShield", "United", "Medicare"]
providers = ["Prov" + str(i) for i in range(1, 7)]
cpt_codes = ["9921" + str(i) for i in range(5)]
dx_codes = ["DX" + str(i).zfill(3) for i in range(8)]

service_start = pd.Timestamp("2023-01-01")
service_dates = service_start + pd.to_timedelta(np.random.randint(0, 180, size=N_CLAIMS), unit="D")
service_dates = np.sort(service_dates)

records = []
for i in range(N_CLAIMS):
    payer = np.random.choice(payers, p=[0.25, 0.25, 0.25, 0.25])
    provider = np.random.choice(providers)
    cpt = np.random.choice(cpt_codes)
    dx = np.random.choice(dx_codes)
    amount = np.round(np.random.gamma(shape=2.0, scale=150.0), 2)
    inpatient = np.random.binomial(1, 0.2)
    age = np.random.randint(18, 90)

    base_risk = 0.15
    base_risk += 0.1 if payer in {"Medicare", "United"} else 0
    base_risk += 0.05 if inpatient else 0
    base_risk += 0.08 if cpt in {"99212", "99213"} else 0
    base_risk += 0.07 if dx in {"DX002", "DX005"} else 0
    base_risk = np.clip(base_risk, 0, 0.85)
    denied = np.random.binomial(1, base_risk)

    records.append(
        {
            "claim_id": f"C{i:04d}",
            "payer": payer,
            "provider": provider,
            "cpt_code": cpt,
            "dx_code": dx,
            "service_date": service_dates[i],
            "claim_amount": float(amount),
            "is_inpatient": inpatient,
            "patient_age": age,
            "denied": denied,
        }
    )

claims = pd.DataFrame(records)
claims = claims.sort_values("service_date").reset_index(drop=True)

output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_claims.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
claims.to_csv(output_path, index=False)

print(f"Wrote {len(claims)} synthetic claims to {output_path}")
