import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "customer_churn_raw.csv"


def poisson_sample(lam: float, rng: random.Random) -> int:
    threshold = math.exp(-lam)
    count = 0
    product = 1.0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def main() -> None:
    rng = random.Random(SEED)
    n_samples = 1500
    tenure = [rng.randrange(1, 72) for _ in range(n_samples)]
    monthly_charges = [round(rng.uniform(20.0, 120.0), 2) for _ in range(n_samples)]

    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(1, n_samples + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": [poisson_sample(1.5, rng) for _ in range(n_samples)],
        "contract_type": rng.choices(
            ["Month-to-Month", "One Year", "Two Year"],
            weights=[0.5, 0.3, 0.2],
            k=n_samples,
        ),
        "internet_service": rng.choices(
            ["DSL", "Fiber Optic", "None"], weights=[0.4, 0.5, 0.1], k=n_samples
        ),
        "churn": rng.choices([0, 1], weights=[0.75, 0.25], k=n_samples),
    }
    df = pd.DataFrame(data)
    df["total_charges"] = df["tenure_months"] * df["monthly_charges"]
    missing_rows = df.sample(frac=0.05, random_state=SEED).index
    df.loc[missing_rows, "total_charges"] = np.nan

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
