"""Controlled CKKS, BGV and BFV benchmark.

Each configuration runs in its own subprocess (fresh OpenFHE global key
state, per-config peak RSS) with a fixed OMP_NUM_THREADS. Per repetition:
new context and keys, WARMUP discarded records, then every test record is
encrypted, evaluated and decrypted with per-operation timing. Correctness is
checked on every repetition and on the two boundary records.

Usage:  uv run python scripts/benchmark.py [--threads 1] [--reps 5] [--sum evalsum|rotations] [--only NAME ...]
Writes results/benchmark_threads<N>_<sum>.json (raw per-record timings are summarised, not stored).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 3
# name -> (scheme, polynomial degree or None)
CONFIGS = {
    "ckks_linear": ("ckks", None),
    "bgv_linear": ("bgv", None),
    "ckks_deg1": ("ckks", 1),
    "bgv_deg1": ("bgv", 1),
    "bfv_linear": ("bfv", None),
    "bfv_deg1": ("bfv", 1),
    "ckks_deg3": ("ckks", 3),
    "ckks_deg5": ("ckks", 5),
}


def summarize(v) -> dict:
    import numpy as np
    a = np.asarray(v) * 1e3
    return {"median_ms": float(np.median(a)), "mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "n": int(a.size)}


def worker(name: str, reps: int, sum_method: str) -> dict:
    import resource

    import numpy as np
    import openfhe as fhe
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

    from fhe_compare import bfv, bgv, ckks
    from fhe_compare.model import load_model, sigmoid
    from fhe_compare.sizes import eval_key_sizes, nbytes

    scheme, degree = CONFIGS[name]
    m = load_model()
    coeffs = m.poly[degree] if degree else None
    rss_base_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if scheme == "ckks":
        lib, depth = ckks, ckks.DEPTH[degree]
        q = None
        prep = lambda x: x
        new_cc = lambda: ckks.make_context(depth)
        model_pts = lambda cc: ckks.encode_model(cc, m.w, m.b)
        run_eval = lambda cc, ct, pts: ckks.evaluate(cc, ct, *pts, coeffs, sum_method)
    else:
        lib = bfv if scheme == "bfv" else bgv
        depth = lib.DEPTH[degree]
        q = lib.Quantizer.build(m.w, m.b, m.clip, m.poly[1])
        prep = q.quantize_x
        new_cc = lambda: lib.make_context(q.t, depth)
        model_pts = lambda cc: lib.encode_model(cc, q)
        run_eval = lambda cc, ct, pts: lib.evaluate(cc, ct, pts, degree, sum_method)

    inputs = [prep(x) for x in m.X_test]
    edge_inputs = [prep(x) for x in m.edge_records()]
    t_ctx, t_keygen, t_enc, t_eval, t_dec, t_total = [], [], [], [], [], []
    outs_by_rep = []
    for rep in range(reps):
        t0 = time.perf_counter(); cc = new_cc(); t1 = time.perf_counter()
        keys = lib.keygen(cc, degree, sum_method); t2 = time.perf_counter()
        t_ctx.append(t1 - t0); t_keygen.append(t2 - t1)
        pts = model_pts(cc)
        for x in inputs[:WARMUP]:
            lib.decrypt(cc, keys.secretKey, run_eval(cc, lib.encrypt(cc, keys.publicKey, x), pts))
        outs = []
        tb = time.perf_counter()
        for x in inputs:
            a = time.perf_counter(); ct = lib.encrypt(cc, keys.publicKey, x)
            b = time.perf_counter(); res = run_eval(cc, ct, pts)
            c = time.perf_counter(); dec = lib.decrypt(cc, keys.secretKey, res)
            d = time.perf_counter()
            t_enc.append(b - a); t_eval.append(c - b); t_dec.append(d - c)
            outs.append(dec)
        t_total.append(time.perf_counter() - tb)
        outs_by_rep.append(outs)
        if rep == 0:
            ct0 = lib.encrypt(cc, keys.publicKey, inputs[0])
            res0 = run_eval(cc, ct0, pts)
            sizes = {"crypto_context": nbytes(cc), "public_key": nbytes(keys.publicKey),
                     "secret_key_client_only": nbytes(keys.secretKey),
                     **eval_key_sizes(cc, relin_generated=(scheme == "ckks" and (degree or 0) >= 2)),
                     "input_ciphertext": nbytes(ct0),
                     "output_ciphertexts": [nbytes(c) for c in res0]}
            params = {"ring_dim": cc.GetRingDimension(), "mult_depth": depth,
                      "batch": 16, "security": "HEStd_128_classic", "slot_sum": sum_method}
            edge_outs = [lib.decrypt(cc, keys.secretKey, run_eval(cc, lib.encrypt(cc, keys.publicKey, x), pts))
                         for x in edge_inputs]
            rss_rep0_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # OpenFHE keeps evaluation keys in library-global maps; drop them so the
        # next repetition does not accumulate the previous repetition's keys.
        del ct, res, pts, keys, cc
        fhe.CryptoContext.ClearStaticMapsAndVectors()

    # correctness
    z_ref = m.z_test
    z_edge = m.edge_records() @ m.w + m.b
    poly = lambda z: np.polynomial.polynomial.polyval(z, coeffs)
    corr = {}
    if scheme == "ckks":
        params.update(scale_bits=ckks.SCALE_BITS, first_mod_bits=ckks.FIRST_MOD_BITS)
        z_he = np.array([[o[0] for o in outs] for outs in outs_by_rep])
        corr["z_abs_err_vs_plain"] = {"mean": float(np.abs(z_he - z_ref).mean()), "max": float(np.abs(z_he - z_ref).max())}
        if coeffs:
            p_he = np.array([[o[1] for o in outs] for outs in outs_by_rep])
            corr["p_abs_err_vs_plain_poly"] = {"mean": float(np.abs(p_he - poly(z_ref)).mean()),
                                               "max": float(np.abs(p_he - poly(z_ref)).max())}
            corr["edge_p_abs_err_vs_plain_poly"] = float(np.abs(np.array([o[1] for o in edge_outs]) - poly(z_edge)).max())
        corr["edge_z_abs_err"] = float(np.abs(np.array([o[0] for o in edge_outs]) - z_edge).max())
        z_last = z_he[-1]
        p_last = p_he[-1] if coeffs else None
    else:
        params.update(plaintext_modulus=q.t, t_bits=q.t.bit_length(), sx=q.sx, sw=q.sw, sc=q.sc,
                      max_abs_zq_over_half_t=q.zmax / q.half,
                      max_abs_pq_bound_over_half_t=(q.pmax / q.half) if degree else None)
        exact = all(o[0] == q.exact_z(x) and (degree is None or o[1] == q.exact_p(q.exact_z(x)))
                    for outs in outs_by_rep for x, o in zip(inputs, outs))
        edge_exact = all(o[0] == q.exact_z(x) and (degree is None or o[1] == q.exact_p(q.exact_z(x)))
                         for x, o in zip(edge_inputs, edge_outs))
        corr["decrypted_equals_exact_integer"] = exact
        corr["edge_decrypted_equals_exact_integer"] = edge_exact
        z_last = np.array([q.dequant_z(o[0]) for o in outs_by_rep[-1]])
        corr["z_quant_err_vs_plain"] = {"mean": float(np.abs(z_last - z_ref).mean()), "max": float(np.abs(z_last - z_ref).max())}
        if degree:
            p_last = np.array([q.dequant_p(o[1]) for o in outs_by_rep[-1]])
            corr["p_quant_err_vs_plain_poly"] = {"mean": float(np.abs(p_last - poly(z_ref)).mean()),
                                                 "max": float(np.abs(p_last - poly(z_ref)).max())}
        else:
            p_last = None
    pred = (z_last > 0).astype(int)
    corr["decision_agreement_with_plain"] = float((pred == (z_ref > 0)).mean())
    score = p_last if p_last is not None else sigmoid(z_last)
    if p_last is not None:
        corr["p_abs_err_vs_sigmoid"] = {"mean": float(np.abs(p_last - sigmoid(z_ref)).mean()),
                                        "max": float(np.abs(p_last - sigmoid(z_ref)).max())}
    y = m.y_test
    metrics = {"accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, score),
               "f1": f1_score(y, pred), "recall": recall_score(y, pred),
               "confusion_matrix": confusion_matrix(y, pred).tolist()}

    lat = np.array(t_enc) + np.array(t_eval) + np.array(t_dec)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "name": name, "scheme": scheme, "degree": degree, "outputs": "z" if degree is None else "z, p(z)",
        "params": params, "sizes_bytes": sizes,
        "timing": {"context": summarize(t_ctx), "keygen": summarize(t_keygen), "encrypt": summarize(t_enc),
                   "evaluate": summarize(t_eval), "decrypt": summarize(t_dec),
                   "latency_per_record": summarize(lat),
                   "throughput_records_per_s_sequential": float(len(inputs) / np.median(t_total))},
        "memory": {"peak_rss_mb_first_rep": rss_rep0_kb / 1024, "peak_rss_mb_all_reps": peak_kb / 1024,
                   "rss_before_crypto_mb": rss_base_kb / 1024,
                   "note": "ru_maxrss high-water mark of the worker process; first_rep = one context, its keys, 60 records"},
        "correctness": corr, "test_metrics_from_decrypted": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS))
    ap.add_argument("--sum", default="evalsum", choices=["evalsum", "rotations"])
    ap.add_argument("--output", type=Path,
                    help="JSON output path (defaults to results/benchmark_threads<N>_<sum>.json)")
    ap.add_argument("--worker")
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(worker(args.worker, args.reps, args.sum)))
        return

    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads)}
    env_info = json.loads((ROOT / "results" / "environment.json").read_text())
    results = []
    for name in args.only or CONFIGS:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, __file__, "--worker", name, "--reps", str(args.reps), "--sum", args.sum],
                              env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"{name} failed (exit {proc.returncode})")
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(r)
        tm, c = r["timing"], r["correctness"]
        ok = c.get("decrypted_equals_exact_integer", True) and c["decision_agreement_with_plain"] == 1.0
        print(f"{name:12s} N={r['params']['ring_dim']} depth={r['params']['mult_depth']} | "
              f"enc {tm['encrypt']['median_ms']:.2f} eval {tm['evaluate']['median_ms']:.2f} "
              f"dec {tm['decrypt']['median_ms']:.2f} ms | keygen {tm['keygen']['median_ms']:.1f} ms | "
              f"peak {r['memory']['peak_rss_mb_first_rep']:.0f}/{r['memory']['peak_rss_mb_all_reps']:.0f} MB | ct {r['sizes_bytes']['input_ciphertext'] / 1024:.0f} KiB "
              f"| agree {c['decision_agreement_with_plain']:.3f} | {'OK' if ok else 'CHECK'} "
              f"({time.perf_counter() - t0:.0f}s)")
    out = args.output or (ROOT / "results" / f"benchmark_threads{args.threads}_{args.sum}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "environment": env_info, "omp_num_threads": args.threads, "slot_sum": args.sum, "repetitions": args.reps,
        "warmup_records_per_rep": WARMUP, "timer": "time.perf_counter, per record, single process",
        "summary_statistic": "median over repetitions x records (keygen/context: over repetitions)",
        "configs": results}, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
