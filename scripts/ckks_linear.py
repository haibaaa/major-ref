"""Phase D step 1: CKKS encrypted linear score z = x . w + b on the test set.

Layout: one record per ciphertext, 11 features in slots 0..10, zeros in
slots 11..15 (batch size 16). The server multiplies by the plaintext
weight vector, sums slots with EvalSum (rotation keys), and adds the bias.
Weights are plaintext to the server; only patient features are encrypted.

Roles are in one process here; client/server separation is Phase G.
"""

import time
from pathlib import Path

import numpy as np
import openfhe as fhe

ROOT = Path(__file__).resolve().parents[1]
BATCH = 16          # power of two >= 11 features
MULT_DEPTH = 1      # one ciphertext-plaintext multiplication
SCALE_BITS = 50
FIRST_MOD_BITS = 60
TOL = 1e-4


def pad(v: np.ndarray) -> np.ndarray:
    out = np.zeros(BATCH)
    out[: len(v)] = v
    return out


def main() -> None:
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    X, w, b, z_ref = art["X_test"], art["w"], float(art["b"]), art["z_test"]
    assert X.shape[1] == len(w) == 11 and BATCH >= len(w)

    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetScalingModSize(SCALE_BITS)
    params.SetFirstModSize(FIRST_MOD_BITS)
    params.SetBatchSize(BATCH)
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(params)
    # ADVANCEDSHE is required by OpenFHE for EvalSumKeyGen / EvalSum
    for f in (fhe.PKESchemeFeature.PKE, fhe.PKESchemeFeature.KEYSWITCH,
              fhe.PKESchemeFeature.LEVELEDSHE, fhe.PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)

    # client: key generation (EvalSum needs rotation keys; no relin key needed for ct*pt)
    t0 = time.perf_counter()
    keys = cc.KeyGen()
    cc.EvalSumKeyGen(keys.secretKey)
    t_keygen = time.perf_counter() - t0

    pt_w = cc.MakeCKKSPackedPlaintext(pad(w))
    pt_b = cc.MakeCKKSPackedPlaintext(np.full(BATCH, b))

    z_he, t_enc, t_eval, t_dec = [], [], [], []
    for i, x in enumerate(X):
        t0 = time.perf_counter()
        ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(pad(x)))      # client
        t1 = time.perf_counter()
        ct_z = cc.EvalAdd(cc.EvalSum(cc.EvalMult(ct, pt_w), BATCH), pt_b)       # server
        t2 = time.perf_counter()
        pt = cc.Decrypt(ct_z, keys.secretKey)                                    # client
        pt.SetLength(BATCH)
        slots = np.array(pt.GetRealPackedValue())
        t3 = time.perf_counter()
        if i == 0:  # verify where EvalSum leaves the result rather than assuming
            print(f"record 0 slots - z_ref (all {BATCH}): max |diff| = {np.abs(slots - z_ref[0]).max():.2e}")
        z_he.append(slots[0])
        t_enc.append(t1 - t0); t_eval.append(t2 - t1); t_dec.append(t3 - t2)

    z_he = np.array(z_he)
    err = np.abs(z_he - z_ref)
    agree = float(((z_he > 0) == (z_ref > 0)).mean())
    ms = lambda v: f"{1e3 * np.mean(v):.2f} ms (median {1e3 * np.median(v):.2f})"
    print(f"CKKS linear | ring dim={cc.GetRingDimension()} | depth={MULT_DEPTH} | scale bits={SCALE_BITS} "
          f"| first mod bits={FIRST_MOD_BITS} | batch={BATCH} | HEStd_128_classic")
    print(f"records={len(X)} | score abs err: mean={err.mean():.2e} max={err.max():.2e} "
          f"| decision agreement with plaintext={agree:.3f}")
    print(f"min |z_ref| (closest to decision boundary) = {np.abs(z_ref).min():.4f}")
    print(f"single-run timings: keygen {1e3 * t_keygen:.1f} ms | per record: enc {ms(t_enc)} "
          f"| eval {ms(t_eval)} | dec {ms(t_dec)}")
    ok = err.max() < TOL and agree == 1.0
    print("CKKS linear score:", "PASS" if ok else f"FAIL (tol {TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
