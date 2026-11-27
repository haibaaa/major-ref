"""Stress-test encrypted logistic decisions at controlled score margins.

Synthetic feature vectors inside the documented clipping box target known
plaintext logits along the weight vector and two orthogonal directions. This
isolates numerical/quantization effects; it is not a patient-data study.

Run: uv run python scripts/margin_stress.py
Writes results/margin_stress.json.
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import openfhe as fhe

from fhe_compare import bfv, bgv, ckks, paillier
from fhe_compare.model import ROOT, load_model

MARGINS = [0.05, 0.02, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
TARGET_SCORES = [-m for m in MARGINS] + [0.0] + list(reversed(MARGINS))
INTEGER_SCALES = {"bgv": [(31, 6, 6), (31, 8, 10), (31, 10, 10), (33, 12, 15)],
                  "bfv": [(31, 6, 6), (31, 8, 10), (31, 10, 10), (33, 12, 15)]}
PAILLIER_BITS = 3072


def stress_records(model):
    """Create bounded x values with controlled logits and varied directions."""
    norm2 = float(model.w @ model.w)
    _, _, vh = np.linalg.svd(model.w.reshape(1, -1), full_matrices=True)
    null_directions = vh[1:3]
    records = []
    for target in TARGET_SCORES:
        base = ((target - model.b) / norm2) * model.w
        vectors = [("aligned", base)]
        vectors.extend((f"orthogonal_{i}", base + 1.5 * direction)
                       for i, direction in enumerate(null_directions))
        for direction, x in vectors:
            if np.max(np.abs(x)) > model.clip:
                raise ValueError(f"target z={target}, direction={direction} exceeds the clip box")
            actual = float(x @ model.w + model.b)
            records.append({"target_z": target, "z_plain": actual,
                            "direction": direction, "x": x})
    return records


def summarize(records: list[dict]) -> dict:
    labeled = [r for r in records if r["decision_plain"] is not None]
    flips = [r for r in labeled if r["decision_observed"] != r["decision_plain"]]
    return {
        "record_count": len(records),
        "labeled_record_count": len(labeled),
        "decision_flips": len(flips),
        "flip_target_scores": [r["target_z"] for r in flips],
        "flip_counts_by_direction": {direction: sum(r["direction"] == direction for r in flips)
                                      for direction in sorted({r["direction"] for r in records})},
        "max_abs_score_error": max(abs(r["z_observed"] - r["z_plain"]) for r in records),
        "median_encrypt_ms": statistics.median(r["encrypt_ms"] for r in records),
        "median_evaluate_ms": statistics.median(r["evaluate_ms"] for r in records),
        "median_decrypt_ms": statistics.median(r["decrypt_ms"] for r in records),
    }


def score_error_bound(weights, clip: float, sx: int, sw: int) -> float:
    """Worst-case score error from nearest-integer x, w and bias quantization."""
    dx, dw = 0.5 / sx, 0.5 / sw
    n = len(weights)
    return (dx * float(np.abs(weights).sum()) + clip * dw * n +
            dx * dw * n + 0.5 / (sx * sw))


def run_ckks(model, records) -> dict:
    cc = ckks.make_context(ckks.DEPTH[None])
    keys = ckks.keygen(cc, None)
    pts = ckks.encode_model(cc, model.w, model.b)
    results = []
    for r in records:
        start = time.perf_counter()
        ct = ckks.encrypt(cc, keys.publicKey, r["x"])
        enc = time.perf_counter()
        out = ckks.evaluate(cc, ct, *pts, None)
        ev = time.perf_counter()
        z = ckks.decrypt(cc, keys.secretKey, out)[0]
        dec = time.perf_counter()
        results.append({"target_z": r["target_z"], "z_plain": r["z_plain"],
                        "direction": r["direction"],
                        "z_observed": z, "decision_plain": r["target_z"] > 0 if r["target_z"] else None,
                        "decision_observed": z > 0,
                        "encrypt_ms": 1e3 * (enc - start),
                        "evaluate_ms": 1e3 * (ev - enc), "decrypt_ms": 1e3 * (dec - ev)})
    fhe.CryptoContext.ClearStaticMapsAndVectors()
    return {"scheme": "ckks", "configuration": {"scale_bits": ckks.SCALE_BITS,
            "first_mod_bits": ckks.FIRST_MOD_BITS}, "summary": summarize(results), "records": results}


def run_integer_scheme(scheme: str, lib, model, records, sx_bits: int, sw_bits: int,
                       t_bits: int, X_train, z_train, X_test, z_test,
                       train_min_margin: float) -> dict:
    q = lib.Quantizer.build(model.w, model.b, model.clip, model.poly[1],
                            t_bits=t_bits, sx_bits=sx_bits, sw_bits=sw_bits)
    cc = lib.make_context(q.t, lib.DEPTH[None])
    keys = lib.keygen(cc, None)
    pts = lib.encode_model(cc, q)
    results = []
    exact = True
    for r in records:
        xq = q.quantize_x(r["x"])
        start = time.perf_counter()
        ct = lib.encrypt(cc, keys.publicKey, xq)
        enc = time.perf_counter()
        out = lib.evaluate(cc, ct, pts, None)
        ev = time.perf_counter()
        raw = lib.decrypt(cc, keys.secretKey, out)[0]
        dec = time.perf_counter()
        exact = exact and raw == q.exact_z(xq)
        z = q.dequant_z(raw)
        results.append({"target_z": r["target_z"], "z_plain": r["z_plain"],
                        "direction": r["direction"],
                        "z_observed": z, "decision_plain": r["target_z"] > 0 if r["target_z"] else None,
                        "decision_observed": z > 0,
                        "encrypt_ms": 1e3 * (enc - start),
                        "evaluate_ms": 1e3 * (ev - enc), "decrypt_ms": 1e3 * (dec - ev)})
    ring_dim = cc.GetRingDimension()
    fhe.CryptoContext.ClearStaticMapsAndVectors()
    error_bound = score_error_bound(model.w, model.clip, q.sx, q.sw)
    train_z_quant = np.array([q.dequant_z(q.exact_z(q.quantize_x(x))) for x in X_train])
    test_z_quant = np.array([q.dequant_z(q.exact_z(q.quantize_x(x))) for x in X_test])
    train_agreement = bool(np.all((train_z_quant > 0) == (z_train > 0)))
    test_agreement = bool(np.all((test_z_quant > 0) == (z_test > 0)))
    observed_max_error = max(float(np.abs(train_z_quant - z_train).max()),
                             float(np.abs(test_z_quant - z_test).max()))
    return {"scheme": scheme, "configuration": {"sx_bits": sx_bits, "sw_bits": sw_bits,
            "t_bits": t_bits, "plaintext_modulus": q.t, "ring_dim": ring_dim,
            "score_error_bound": error_bound,
            "train_min_abs_margin": train_min_margin,
            "bound_below_train_min_margin": error_bound < train_min_margin,
            "training_decision_agreement": train_agreement,
            "test_decision_agreement": test_agreement,
            "max_abs_error_train_and_test": observed_max_error,
            "error_bound_covers_train_and_test": observed_max_error <= error_bound,
            "linear_bound_fraction_of_t_half": q.zmax / q.half},
            "decrypted_equals_exact_integer": exact,
            "summary": summarize(results), "records": results}


def run_paillier(model, records) -> dict:
    public, private = paillier.keygen(PAILLIER_BITS)
    results = []
    for r in records:
        start = time.perf_counter()
        encrypted = paillier.encrypt(public, r["x"].tolist())
        enc = time.perf_counter()
        output = paillier.evaluate(encrypted, model.w.tolist(), model.b)
        ev = time.perf_counter()
        z = paillier.decrypt(private, output)[0]
        dec = time.perf_counter()
        results.append({"target_z": r["target_z"], "z_plain": r["z_plain"],
                        "direction": r["direction"],
                        "z_observed": z, "decision_plain": r["target_z"] > 0 if r["target_z"] else None,
                        "decision_observed": z > 0,
                        "encrypt_ms": 1e3 * (enc - start),
                        "evaluate_ms": 1e3 * (ev - enc), "decrypt_ms": 1e3 * (dec - ev)})
    return {"scheme": "paillier", "configuration": {"modulus_bits": PAILLIER_BITS},
            "summary": summarize(results), "records": results}


def main() -> None:
    model = load_model()
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    X_train, z_train = art["X_train"], art["z_train"]
    train_min_margin = float(np.abs(z_train).min())
    records = stress_records(model)
    configurations = []
    configurations.append(run_ckks(model, records))
    for name, lib in (("bgv", bgv), ("bfv", bfv)):
        for t_bits, sx_bits, sw_bits in INTEGER_SCALES[name]:
            configurations.append(run_integer_scheme(name, lib, model, records,
                sx_bits, sw_bits, t_bits, X_train, z_train, model.X_test,
                model.z_test, train_min_margin))
    configurations.append(run_paillier(model, [r for r in records if r["direction"] == "aligned"]))
    data = {
        "experiment": "synthetic controlled-logit margin stress test",
        "method": "construct x on the target hyperplane, then add 0 or 1.5-unit perturbations in two directions orthogonal to w; all vectors remain in the model clip box",
        "target_scores": TARGET_SCORES,
        "training_min_abs_margin": train_min_margin,
        "decision_rule": "z > 0; exact target z=0 is an unlabeled tie, excluded from flip counts",
        "limitations": ["synthetic vectors isolate arithmetic error and are not representative patient records",
                        "one cryptographic run per record/configuration; timings are exploratory",
                        "Paillier uses 3072-bit modulus; decimal size comparison is outside this experiment"],
        "configurations": configurations,
    }
    out = ROOT / "results" / "margin_stress.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    for c in configurations:
        s = c["summary"]
        print(f"{c['scheme']:8s} {c['configuration']}: flips={s['decision_flips']}/{s['labeled_record_count']} "
              f"max_error={s['max_abs_score_error']:.2e} "
              f"median latency={s['median_encrypt_ms'] + s['median_evaluate_ms'] + s['median_decrypt_ms']:.2f} ms")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
