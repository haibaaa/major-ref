"""Phase C smoke test: minimal CKKS context, encrypt, multiply, decrypt.

Checks that the OpenFHE native extension works and records the observed
approximation error. This is not the inference circuit.
"""

import numpy as np
import openfhe as fhe

MULT_DEPTH = 2        # enough for (x*y)*y
SCALE_MOD_BITS = 50   # scaling-factor bits
BATCH_SIZE = 8
TOL = 1e-6


def main() -> None:
    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetScalingModSize(SCALE_MOD_BITS)
    params.SetBatchSize(BATCH_SIZE)
    params.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)  # relinearization key

    x = [0.5, -1.25, 2.0, 3.14159, -0.75, 1.0, 0.0, -2.5]
    y = [1.5, 0.8, -1.0, 0.5, 2.0, -3.0, 4.0, 0.1]

    ct_x = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
    ct_y = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(y))

    ct_xy = cc.EvalMult(ct_x, ct_y)           # depth 1
    ct_xyy = cc.EvalMult(ct_xy, ct_y)         # depth 2
    ct_sum = cc.EvalAdd(ct_x, ct_y)

    def dec(ct) -> np.ndarray:
        pt = cc.Decrypt(ct, keys.secretKey)
        pt.SetLength(BATCH_SIZE)
        return np.array(pt.GetRealPackedValue())

    xa, ya = np.array(x), np.array(y)
    checks = {
        "x+y": (dec(ct_sum), xa + ya),
        "x*y": (dec(ct_xy), xa * ya),
        "x*y*y": (dec(ct_xyy), xa * ya * ya),
    }

    print(f"OpenFHE CKKS | ring dim={cc.GetRingDimension()} "
          f"| depth={MULT_DEPTH} | scale bits={SCALE_MOD_BITS} "
          f"| security=HEStd_128_classic")
    ok = True
    for name, (got, want) in checks.items():
        err = np.max(np.abs(got - want))
        ok &= err < TOL
        print(f"  {name:6s} max abs err = {err:.3e}")
    print("CKKS smoke test:", "PASS" if ok else f"FAIL (tol {TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
