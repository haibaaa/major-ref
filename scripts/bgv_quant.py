"""Phase E step 1: BGV integer-quantization readiness (no encryption).

Scheme (all integer arithmetic done with exact Python ints):
    x_q = round(x * Sx),  w_q = round(w * Sw),  b_q = round(b * Sx * Sw)
    z_q = sum(x_q * w_q) + b_q                   scale S = Sx * Sw
    P_q = sum_k round(c_k * Sc) * z_q^k * S^(d-k)  scale Sc * S^d

Bounds:
  * linear: features are clipped to [-clip, clip], so |x_q| <= clip*Sx and the
    exact reachable maximum is |z_q| <= clip*Sx*sum|w_q| + |b_q|.
  * polynomial: triangle-inequality bound sum |a_k| Zmax^k S^(d-k)
    (sufficient, not necessarily tight).
A centered plaintext modulus t holds values in [-(t-1)/2, (t-1)/2], so we need
2*bound + 1 <= t, i.e. bits(t) > log2(2*bound + 1).
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SX_BITS = [4, 6, 8, 10]
SW_BITS = [4, 6, 8, 10]
SC_BITS = 8


def quantize(art: dict, meta: dict, sx: int, sw: int):
    X = art["X_test"]
    xq = [[round(v * sx) for v in row] for row in X]
    wq = [round(v * sw) for v in art["w"]]
    bq = round(float(art["b"]) * sx * sw)
    zq = [sum(a * c for a, c in zip(row, wq)) + bq for row in xq]
    clip = meta["preprocessing"]["clip"][1]
    zmax = int(clip * sx) * sum(abs(v) for v in wq) + abs(bq)   # exact reachable max
    return xq, wq, bq, zq, zmax


def poly_int(zq: list[int], coeffs: list[float], s: int, sc: int) -> list[int]:
    d = len(coeffs) - 1
    a = [round(c * sc) for c in coeffs]
    return [sum(a[k] * z**k * s ** (d - k) for k in range(d + 1)) for z in zq]


def poly_bound(zmax: int, coeffs: list[float], s: int, sc: int) -> int:
    d = len(coeffs) - 1
    return sum(abs(round(c * sc)) * zmax**k * s ** (d - k) for k, c in enumerate(coeffs))


def bits(bound: int) -> float:
    return math.log2(2 * bound + 1)


def main() -> None:
    art = dict(np.load(ROOT / "artifacts" / "hf_plain.npz"))
    meta = json.loads((ROOT / "artifacts" / "hf_meta.json").read_text())
    z_ref = art["z_test"]
    polys = meta["polynomial"]["degrees"]
    sc = 2**SC_BITS

    print(f"Sc = 2^{SC_BITS} | test n={len(z_ref)} | min |z| = {np.abs(z_ref).min():.4f}")
    print(f"{'Sx':>5} {'Sw':>5} | {'z max err':>9} {'agree':>5} {'lin bits':>8} |"
          + "".join(f" {'d' + k + ' err':>8} {'d' + k + ' bits':>8} |" for k in polys))
    rows = []
    for bx in SX_BITS:
        for bw in SW_BITS:
            sx, sw = 2**bx, 2**bw
            s = sx * sw
            _, _, _, zq, zmax = quantize(art, meta, sx, sw)
            z_deq = np.array(zq, dtype=float) / s
            row = {"sx_bits": bx, "sw_bits": bw, "z_max_err": float(np.abs(z_deq - z_ref).max()),
                   "agree": float(((np.array(zq) > 0) == (z_ref > 0)).mean()),
                   "linear_bound": zmax, "linear_bits": bits(zmax)}
            line = f"2^{bx:<3} 2^{bw:<3} | {row['z_max_err']:9.2e} {row['agree']:5.3f} {row['linear_bits']:8.1f} |"
            for k, info in polys.items():
                c = info["coefficients_ascending"]
                pq = poly_int(zq, c, s, sc)
                p_deq = np.array([v / (sc * s ** (len(c) - 1)) for v in pq])
                p_ref = np.polynomial.polynomial.polyval(z_ref, c)
                row[f"d{k}_err"] = float(np.abs(p_deq - p_ref).max())
                row[f"d{k}_bits"] = bits(poly_bound(zmax, c, s, sc))
                line += f" {row[f'd{k}_err']:8.1e} {row[f'd{k}_bits']:8.1f} |"
            rows.append(row)
            print(line)
    out = ROOT / "results" / "bgv_quant_sweep.json"
    out.write_text(json.dumps({"sc_bits": SC_BITS, "rows": rows}, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
