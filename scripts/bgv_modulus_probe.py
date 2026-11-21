"""Phase E: empirically find the largest usable BGV plaintext modulus t.

For each scaling technique and t size, build a 128-bit-security BGV context
(depth 3), multiply a value near t/2 down a depth-3 chain, and check the
decryption against exact arithmetic mod t. Context creation alone is not
taken as evidence: some settings build a context but decrypt incorrectly.

t is the smallest prime >= 2^(bits-1) with t = 1 (mod 2^16), which allows SIMD
batching for ring dimensions up to 2^15.
Writes results/bgv_modulus_probe.json.
"""

import json
from pathlib import Path

import openfhe as fhe

ROOT = Path(__file__).resolve().parents[1]
DEPTH = 3
REPEATS = 3
ST, F = fhe.ScalingTechnique, fhe.PKESchemeFeature
SETTINGS = {
    "FLEXIBLEAUTO (default)": dict(tech=ST.FLEXIBLEAUTO, mod_bits=None),
    "FIXEDMANUAL, 60-bit moduli": dict(tech=ST.FIXEDMANUAL, mod_bits=60),
}
T_BITS = [30, 31, 32, 33, 34, 36, 40]


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def batching_prime(bits: int, m: int = 1 << 16) -> int:
    k = (1 << (bits - 1)) // m + 1
    while not is_prime(k * m + 1):
        k += 1
    return k * m + 1


def trial(t: int, tech, mod_bits) -> dict:
    p = fhe.CCParamsBGVRNS()
    p.SetPlaintextModulus(t)
    p.SetMultiplicativeDepth(DEPTH)
    p.SetScalingTechnique(tech)
    if mod_bits:
        p.SetScalingModSize(mod_bits)
        p.SetFirstModSize(mod_bits)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(p)
    for f in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE):
        cc.Enable(f)
    k = cc.KeyGen()
    cc.EvalMultKeyGen(k.secretKey)
    half = (t - 1) // 2
    y = 3
    x = half // y**DEPTH                       # x * y^DEPTH lands just below t/2
    ct = cc.Encrypt(k.publicKey, cc.MakePackedPlaintext([x, -x, 12345]))
    cy = cc.Encrypt(k.publicKey, cc.MakePackedPlaintext([y, y, 2]))
    for _ in range(DEPTH):
        ct = cc.EvalMult(ct, cy)
        if tech == ST.FIXEDMANUAL:
            ct = cc.ModReduce(ct)
    pt = cc.Decrypt(ct, k.secretKey)
    pt.SetLength(3)
    want = [x * y**DEPTH, -x * y**DEPTH, 12345 * 2**DEPTH]
    return {"ring_dim": cc.GetRingDimension(), "correct": list(pt.GetPackedValue()) == want}


def main() -> None:
    results = []
    for name, s in SETTINGS.items():
        for bits in T_BITS:
            t = batching_prime(bits)
            row = {"setting": name, "t_bits": t.bit_length(), "t": t}
            try:
                runs = [trial(t, s["tech"], s["mod_bits"]) for _ in range(REPEATS)]
                row.update(ring_dim=runs[0]["ring_dim"], correct_runs=sum(r["correct"] for r in runs),
                           repeats=REPEATS)
                status = f"N={row['ring_dim']} correct {row['correct_runs']}/{REPEATS}"
            except RuntimeError as e:
                row["error"] = str(e).splitlines()[0][-160:]
                status = "context ERROR: " + row["error"][-90:]
            results.append(row)
            print(f"{name:28s} t bits={row['t_bits']}: {status}")
    out = ROOT / "results" / "bgv_modulus_probe.json"
    out.write_text(json.dumps({"depth": DEPTH, "security": "HEStd_128_classic", "rows": results}, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
