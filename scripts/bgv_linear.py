"""Phase E: BGV encrypted linear score and degree-1 sigmoid on the test set.

Quantization (client): x_q = round(x*Sx); model (server): w_q = round(w*Sw),
b_q = round(b*Sx*Sw), degree-1 coefficients a_k = round(c_k*Sc).
Circuit (server), one record per ciphertext, slots 0..10, batch 16:
    z_q = EvalSum(x_q * w_q) + b_q                          scale S = Sx*Sw
    P_q = a1 * z_q + a0 * S                                 scale Sc*S
The client decrypts, checks values against exact Python-int arithmetic,
and dequantizes: z = z_q / S, p = P_q / (Sc*S). Decision is taken from z
(sign), p is only a score (see CKKS boundary findings).

Wraparound safety: features are clipped, so |x_q| <= clip*Sx and the exact
reachable maxima of |z_q| and |P_q| are checked against (t-1)/2 before
encrypting anything. Boundary records attaining those maxima are run too.
Degrees 3 and 5 are out of scope at a single modulus; see bgv_quant.py.
"""

import json
import time
from pathlib import Path

import numpy as np
import openfhe as fhe

from bgv_modulus_probe import batching_prime

ROOT = Path(__file__).resolve().parents[1]
BATCH = 16
SX, SW = 2**8, 2**10
T_BITS = 31
MULT_DEPTH = 2
F = fhe.PKESchemeFeature


def pad(v: list[int]) -> list[int]:
    return list(v) + [0] * (BATCH - len(v))


def main() -> None:
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    meta = json.loads((ROOT / "artifacts" / "hf_meta.json").read_text())
    X, w, b, z_ref = art["X_test"], art["w"], float(art["b"]), art["z_test"]
    clip = meta["preprocessing"]["clip"][1]
    c0, c1 = meta["polynomial"]["degrees"]["1"]["coefficients_ascending"]

    t = batching_prime(T_BITS)
    half = (t - 1) // 2
    S = SX * SW
    w_q = [round(v * SW) for v in w]
    b_q = round(b * S)

    # exact worst-case bounds over the clip box (|x_q| <= clip*Sx, integer)
    xmax = round(clip * SX)
    zmax = xmax * sum(abs(v) for v in w_q) + abs(b_q)
    # largest Sc whose degree-1 bound fits in the centered range
    sc_bits = max(k for k in range(1, 30)
                  if abs(round(c1 * 2**k)) * zmax + abs(round(c0 * 2**k)) * S <= half)
    SC = 2**sc_bits
    a0, a1 = round(c0 * SC), round(c1 * SC)
    pmax = abs(a1) * zmax + abs(a0) * S
    print(f"t={t} ({t.bit_length()} bits, (t-1) % 2^16 == 0) | half={half}")
    print(f"Sx=2^8 Sw=2^10 Sc=2^{sc_bits} | max|z_q|={zmax} ({zmax / half:.3f} of t/2) "
          f"| max|P_q|={pmax} ({pmax / half:.3f} of t/2)")
    assert zmax <= half and pmax <= half

    # client-side quantization (exact ints); boundary records attain |x_q| = xmax
    sgn = np.sign(w).astype(int)
    X_q = [[round(v * SX) for v in row] for row in X]
    edge = [[int(s) * xmax for s in sgn], [-int(s) * xmax for s in sgn]]
    exact_z = lambda xq: sum(a * c for a, c in zip(xq, w_q)) + b_q
    exact_p = lambda zq: a1 * zq + a0 * S

    params = fhe.CCParamsBGVRNS()
    params.SetPlaintextModulus(t)
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetBatchSize(BATCH)
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(params)
    for f in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE, F.ADVANCEDSHE):
        cc.Enable(f)
    assert cc.GetPlaintextModulus() == t

    t0 = time.perf_counter()
    keys = cc.KeyGen()
    cc.EvalSumKeyGen(keys.secretKey)   # rotation keys for EvalSum; ct*pt needs no relin key
    t_keygen = time.perf_counter() - t0

    pt_w = cc.MakePackedPlaintext(pad(w_q))
    pt_b = cc.MakePackedPlaintext([b_q] * BATCH)
    pt_a1 = cc.MakePackedPlaintext([a1] * BATCH)
    pt_a0S = cc.MakePackedPlaintext([a0 * S] * BATCH)

    def infer(xq):
        t0 = time.perf_counter()
        ct = cc.Encrypt(keys.publicKey, cc.MakePackedPlaintext(pad(xq)))            # client
        t1 = time.perf_counter()
        ct_z = cc.EvalAdd(cc.EvalSum(cc.EvalMult(ct, pt_w), BATCH), pt_b)           # server
        ct_p = cc.EvalAdd(cc.EvalMult(ct_z, pt_a1), pt_a0S)
        t2 = time.perf_counter()
        out = []
        for c in (ct_z, ct_p):                                                       # client
            pt = cc.Decrypt(c, keys.secretKey)
            pt.SetLength(BATCH)
            out.append(list(pt.GetPackedValue()))
        t3 = time.perf_counter()
        return out, (t1 - t0, t2 - t1, t3 - t2)

    # Observed for BGV in OpenFHE 1.6.1: EvalSum leaves suffix sums (slot i = sum of
    # slots i..BATCH-1), unlike CKKS where every slot held the total. Only slot 0 is used.
    (z_slots, _), _ = infer(X_q[0])
    print(f"record 0: slot 0 == exact z_q: {z_slots[0] == exact_z(X_q[0])} "
          f"| all slots equal: {len(set(z_slots)) == 1} (BGV EvalSum gives suffix sums)")

    zq_he, pq_he, times = [], [], []
    for xq in X_q:
        (zs, ps), tm = infer(xq)
        zq_he.append(zs[0]); pq_he.append(ps[0]); times.append(tm)
    exact_ok = all(z == exact_z(xq) and p == exact_p(exact_z(xq)) for xq, z, p in zip(X_q, zq_he, pq_he))

    z_deq = np.array(zq_he, dtype=float) / S
    p_deq = np.array(pq_he, dtype=float) / (SC * S)
    p_ref = c0 + c1 * z_ref
    agree = float(((np.array(zq_he) > 0) == (z_ref > 0)).mean())
    print(f"BGV | ring dim={cc.GetRingDimension()} | depth={MULT_DEPTH} | batch={BATCH} | HEStd_128_classic")
    print(f"records={len(X_q)} | decrypted == exact integer result: {exact_ok}")
    print(f"z quantization err: mean={np.abs(z_deq - z_ref).mean():.2e} max={np.abs(z_deq - z_ref).max():.2e} "
          f"| decision agreement with plaintext={agree:.3f}")
    print(f"deg-1 p quantization err vs float poly: mean={np.abs(p_deq - p_ref).mean():.2e} "
          f"max={np.abs(p_deq - p_ref).max():.2e}")

    edge_ok = []
    for xq in edge:
        (zs, ps), _ = infer(xq)
        ez, ep = exact_z(xq), exact_p(exact_z(xq))
        edge_ok.append(zs[0] == ez and ps[0] == ep)
        print(f"boundary: z_q dec={zs[0]} exact={ez} ({abs(ez) / half:.3f} of t/2) "
              f"| P_q dec={ps[0]} exact={ep} ({abs(ep) / half:.3f} of t/2)")

    tt = np.array(times) * 1e3
    print(f"single-run timings: keygen {1e3 * t_keygen:.1f} ms | median per record: enc {np.median(tt[:, 0]):.2f} "
          f"| eval {np.median(tt[:, 1]):.2f} | dec (2 ciphertexts) {np.median(tt[:, 2]):.2f} ms")
    ok = exact_ok and all(edge_ok) and agree == 1.0
    print("BGV linear/degree-1:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
