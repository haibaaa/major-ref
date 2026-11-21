"""Phase C smoke test: minimal BGV context with small integers.

Verifies encode/encrypt/add/multiply/decrypt and that results agree with
exact arithmetic reduced modulo the plaintext modulus t (centered
representation, as returned by OpenFHE packed encoding). Includes one
deliberate wraparound case to show BGV is NOT unbounded integer arithmetic.
"""

import openfhe as fhe

PLAINTEXT_MODULUS = 65537  # prime, 65537 = 2^16 + 1 => t ≡ 1 (mod 2N) for N <= 2^15
MULT_DEPTH = 2


def centered_mod(v: int, t: int) -> int:
    """Map v into OpenFHE's centered range [-(t-1)/2, (t-1)/2]."""
    r = v % t
    return r - t if r > t // 2 else r


def main() -> None:
    params = fhe.CCParamsBGVRNS()
    params.SetPlaintextModulus(PLAINTEXT_MODULUS)
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)

    n = cc.GetRingDimension()
    t = cc.GetPlaintextModulus()
    assert (t - 1) % (2 * n) == 0, f"t={t} not batching-compatible for N={n}"

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # last slot pair (300 * 300 = 90000) intentionally exceeds t/2 to wrap
    x = [1, 2, 3, -4, 5, -6, 7, 300]
    y = [10, -20, 30, 40, 0, 6, -7, 300]
    k = len(x)

    ct_x = cc.Encrypt(keys.publicKey, cc.MakePackedPlaintext(x))
    ct_y = cc.Encrypt(keys.publicKey, cc.MakePackedPlaintext(y))
    ct_sum = cc.EvalAdd(ct_x, ct_y)
    ct_xy = cc.EvalMult(ct_x, ct_y)
    ct_xyy = cc.EvalMult(ct_xy, ct_y)

    def dec(ct) -> list[int]:
        pt = cc.Decrypt(ct, keys.secretKey)
        pt.SetLength(k)
        return list(pt.GetPackedValue())

    exact = {
        "x+y": [a + b for a, b in zip(x, y)],
        "x*y": [a * b for a, b in zip(x, y)],
        "x*y*y": [a * b * b for a, b in zip(x, y)],
    }
    got = {"x+y": dec(ct_sum), "x*y": dec(ct_xy), "x*y*y": dec(ct_xyy)}

    print(f"OpenFHE BGV | ring dim={n} | t={t} | depth={MULT_DEPTH} "
          f"| security=HEStd_128_classic | (t-1)%2N==0: True")
    ok = True
    for name in exact:
        want_mod = [centered_mod(v, t) for v in exact[name]]
        match = got[name] == want_mod
        wrapped = [i for i, (e, w) in enumerate(zip(exact[name], want_mod)) if e != w]
        ok &= match
        print(f"  {name:6s} matches exact mod t: {match} | wrapped slots: {wrapped}")
    print(f"  wrap demo: 300*300 exact=90000, decrypted={got['x*y'][-1]} "
          f"(= 90000 mod {t}, centered)")
    print("BGV smoke test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
