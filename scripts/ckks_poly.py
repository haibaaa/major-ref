"""Phase D step 2: CKKS encrypted logistic score with polynomial sigmoid.

Circuit: z = EvalSum(x * w) + b  ->  p(z) = EvalPoly(z, coeffs), coeffs from
artifacts/hf_meta.json (degrees 1, 3, 5). For each degree the minimum
multiplicative depth is found empirically (smallest depth whose decrypted
output matches the plaintext polynomial), then the full test set is run.

Two errors are reported separately:
  * CKKS error:            |dec(p(z)) - p(z_ref)|   (encryption arithmetic)
  * approximation error:   |p(z_ref) - sigmoid(z_ref)|  (polynomial choice)

A boundary test evaluates synthetic worst-case records (all features at the
clip limit, aligned with sign(w)) to show the polynomial outside its fit range.
"""

import json
import time
from pathlib import Path

import numpy as np
import openfhe as fhe

ROOT = Path(__file__).resolve().parents[1]
BATCH = 16
SCALE_BITS = 50
FIRST_MOD_BITS = 60
TOL = 1e-4
MAX_DEPTH_TRY = 8
F = fhe.PKESchemeFeature


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pad(v: np.ndarray) -> np.ndarray:
    out = np.zeros(BATCH)
    out[: len(v)] = v
    return out


def make_context(depth: int):
    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(SCALE_BITS)
    params.SetFirstModSize(FIRST_MOD_BITS)
    params.SetBatchSize(BATCH)
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(params)
    for f in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE, F.ADVANCEDSHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)   # relinearization (ct*ct inside EvalPoly)
    cc.EvalSumKeyGen(keys.secretKey)    # rotations for EvalSum
    return cc, keys


def run(cc, keys, X, w, b, coeffs):
    """Encrypt each record, evaluate the circuit, decrypt slot 0. Returns outputs and timings."""
    pt_w = cc.MakeCKKSPackedPlaintext(pad(w))
    pt_b = cc.MakeCKKSPackedPlaintext(np.full(BATCH, b))
    out, t_enc, t_eval, t_dec = [], [], [], []
    for x in X:
        t0 = time.perf_counter()
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(pad(x)))
        t1 = time.perf_counter()
        ct_z = cc.EvalAdd(cc.EvalSum(cc.EvalMult(ct, pt_w), BATCH), pt_b)
        ct_p = cc.EvalPoly(ct_z, coeffs)
        t2 = time.perf_counter()
        pt = cc.Decrypt(ct_p, keys.secretKey)
        pt.SetLength(1)
        out.append(pt.GetRealPackedValue()[0])
        t3 = time.perf_counter()
        t_enc.append(t1 - t0); t_eval.append(t2 - t1); t_dec.append(t3 - t2)
    return np.array(out), (t_enc, t_eval, t_dec)


def main() -> None:
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    meta = json.loads((ROOT / "artifacts" / "hf_meta.json").read_text())
    X, w, b, z_ref = art["X_test"], art["w"], float(art["b"]), art["z_test"]
    # artifacts must describe the same model
    assert np.allclose(w, meta["model"]["weights"]) and np.isclose(b, meta["model"]["bias"])
    clip = meta["preprocessing"]["clip"][1]
    r = meta["polynomial"]["fit_range"][1]

    # synthetic boundary records: every feature at +/-clip aligned with sign(w)
    X_edge = np.stack([clip * np.sign(w), -clip * np.sign(w)])
    z_edge = X_edge @ w + b

    print(f"fit range +/-{r:.2f} | test z in [{z_ref.min():.2f}, {z_ref.max():.2f}] "
          f"| boundary z = {np.round(z_edge, 2).tolist()}")
    for deg, info in meta["polynomial"]["degrees"].items():
        coeffs = [float(c) for c in info["coefficients_ascending"]]
        p_ref = np.polynomial.polynomial.polyval(z_ref, coeffs)

        # find minimal depth on a few records
        depth = None
        for d in range(1, MAX_DEPTH_TRY + 1):
            try:
                cc, keys = make_context(d)
                got, _ = run(cc, keys, X[:3], w, b, coeffs)
            except RuntimeError as e:
                print(f"  degree {deg}: depth {d} -> error: {str(e).splitlines()[0][-120:]}")
                continue
            if np.abs(got - p_ref[:3]).max() < TOL:
                depth = d
                break
            print(f"  degree {deg}: depth {d} -> wrong result (max err {np.abs(got - p_ref[:3]).max():.2e})")
        if depth is None:
            print(f"degree {deg}: no depth <= {MAX_DEPTH_TRY} produced a correct result")
            continue

        t0 = time.perf_counter()
        cc, keys = make_context(depth)
        t_setup = time.perf_counter() - t0
        p_he, (t_enc, t_eval, t_dec) = run(cc, keys, X, w, b, coeffs)
        ckks_err = np.abs(p_he - p_ref)
        approx_err = np.abs(p_ref - sigmoid(z_ref))
        agree = float(((p_he > 0.5) == (z_ref > 0)).mean())
        ms = lambda v: f"{1e3 * np.median(v):.2f}"
        print(f"degree {deg}: min depth={depth} | ring dim={cc.GetRingDimension()} "
              f"| context+keys {1e3 * t_setup:.0f} ms")
        print(f"   CKKS err vs plaintext poly: mean={ckks_err.mean():.2e} max={ckks_err.max():.2e}")
        print(f"   poly err vs sigmoid (test): mean={approx_err.mean():.4f} max={approx_err.max():.4f}")
        print(f"   decision agreement with plaintext LR={agree:.3f} "
              f"| output range [{p_he.min():.3f}, {p_he.max():.3f}]")
        print(f"   median per record (ms): enc {ms(t_enc)} | eval {ms(t_eval)} | dec {ms(t_dec)}")

        p_edge_he, _ = run(cc, keys, X_edge, w, b, coeffs)
        p_edge = np.polynomial.polynomial.polyval(z_edge, coeffs)
        print(f"   boundary: dec={np.round(p_edge_he, 3).tolist()} plaintext poly={np.round(p_edge, 3).tolist()} "
              f"sigmoid={np.round(sigmoid(z_edge), 3).tolist()}")


if __name__ == "__main__":
    main()
