"""Measure Paillier as an additive-encryption linear-score baseline.

Unlike CKKS/BGV/BFV, Paillier encrypts each feature separately and does not
support the higher-degree polynomial circuits. Example:
    uv run python scripts/benchmark_paillier.py --reps 3 --key-bits 3072 --records 5
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from fhe_compare import paillier
from fhe_compare.model import ROOT, load_model


def summary(samples: list[float]) -> dict:
    ms = [1000 * x for x in samples]
    return {"median_ms": statistics.median(ms),
            "mean_ms": statistics.mean(ms),
            "p95_ms": float(np.percentile(ms, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--key-bits", type=int, default=3072,
                        help="modulus length; 3072 is a conventional ~128-bit-security comparison point")
    parser.add_argument("--records", type=int, default=None,
                        help="number of deterministic, evenly spaced test records (default: all)")
    parser.add_argument("--output", type=Path,
                        help="output JSON (default: filename includes key size and sample size)")
    args = parser.parse_args()
    if args.reps < 1 or (args.records is not None and args.records < 1):
        parser.error("--reps and --records must be positive")

    model = load_model()
    record_count = min(args.records or len(model.X_test), len(model.X_test))
    record_ids = np.linspace(0, len(model.X_test) - 1, record_count, dtype=int)
    X, z_ref = model.X_test[record_ids], model.z_test[record_ids]
    enc_times, eval_times, dec_times, key_times = [], [], [], []
    errors, agreements, request_bytes, response_bytes, public_sizes = [], [], [], [], []
    for _ in range(args.reps):
        start = time.perf_counter()
        public_key, private_key = paillier.keygen(args.key_bits)
        key_times.append(time.perf_counter() - start)
        print(f"rep {len(key_times)}/{args.reps}: keygen {key_times[-1]:.2f}s", file=sys.stderr, flush=True)
        public_sizes.append(paillier.public_key_bytes(public_key))
        predicted = []
        for x, z in zip(X, z_ref):
            start = time.perf_counter()
            encrypted = paillier.encrypt(public_key, x.tolist())
            enc_times.append(time.perf_counter() - start)
            request_bytes.append(sum(paillier.ciphertext_bytes(ct) for ct in encrypted))
            start = time.perf_counter()
            encrypted_z = paillier.evaluate(encrypted, model.w.tolist(), model.b)
            eval_times.append(time.perf_counter() - start)
            response_bytes.append(sum(paillier.ciphertext_bytes(ct) for ct in encrypted_z))
            start = time.perf_counter()
            z_he = paillier.decrypt(private_key, encrypted_z)[0]
            dec_times.append(time.perf_counter() - start)
            errors.append(abs(z_he - z))
            predicted.append(z_he > 0)
        agreements.append(float(np.mean(np.asarray(predicted) == (z_ref > 0))))

    data = {
        "scheme": "Paillier",
        "scope": "linear score only; model weights plaintext at evaluator",
        "key_bits": args.key_bits,
        "security_note": ("Functional smoke only: this key size is not secure or comparable to HEStd_128_classic."
                          if args.key_bits < 3072 else
                          "3072-bit modulus is a conventional approximate 128-bit comparison point; not a formal parameter equivalence."),
        "repetitions": args.reps,
        "records_per_repetition": len(X),
        "record_sampling": "deterministically evenly spaced from the fixed test set",
        "timing": {"keygen": summary(key_times), "encrypt_record": summary(enc_times),
                   "evaluate_record": summary(eval_times), "decrypt_score": summary(dec_times)},
        "correctness": {"max_abs_score_error": max(errors),
                        "mean_abs_score_error": statistics.mean(errors),
                        "decision_agreement": statistics.mean(agreements)},
        "approx_decimal_wire_bytes": {"public_key": statistics.median(public_sizes),
                                      "request_per_record": statistics.median(request_bytes),
                                      "response_per_record": statistics.median(response_bytes),
                                      "note": "decimal ciphertext representation plus 8-byte exponent estimate; not a binary wire protocol"},
    }
    default_name = f"benchmark_paillier_{args.key_bits}b_{len(X)}records.json"
    out = args.output or (ROOT / "results" / default_name)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
