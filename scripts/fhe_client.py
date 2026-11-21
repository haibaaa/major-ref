"""CLIENT command line.

    uv run python scripts/fhe_client.py setup   --run demo/ckks_deg1
    uv run python scripts/fhe_client.py encrypt --run demo/ckks_deg1
    uv run python scripts/fhe_client.py decrypt --run demo/ckks_deg1

`encrypt` takes the held-out test records (raw CSV rows) as the client's
patient records and preprocesses them locally with the published parameters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fhe_compare import client
from fhe_compare.model import ROOT
from fhe_compare.protocol import Layout

CSV = ROOT / "data" / "Heart_failure_clinical_records_dataset.csv"


def test_records() -> list[dict[str, float]]:
    idx = np.load(ROOT / "artifacts" / "hf_plain.npz")["idx_test"]
    return pd.read_csv(CSV).iloc[idx].to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["setup", "encrypt", "decrypt"])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    layout = Layout(args.run)
    if args.step == "setup":
        out = client.setup(layout)
    elif args.step == "encrypt":
        out = client.encrypt(layout, test_records())
    else:
        res = client.decrypt(layout)
        out = {"responses": len(res), "positive_decisions": sum(r["decision"] for r in res)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
