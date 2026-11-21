"""CKKS encrypted logistic-regression inference.

Layout: one record per ciphertext; features in slots 0..10, zeros in 11..15.
Circuit (SERVER): z = slot_sum(ct * w) + b, optionally p = EvalPoly(z, coeffs).
The server returns z and, when a polynomial is requested, p as separate
ciphertexts; the client takes the decision from sign(z) and uses p as a score.

Minimum multiplicative depths below were measured by scripts/ckks_poly.py
(OpenFHE 1.6.1, scale 50 bits, first modulus 60 bits).
"""

import numpy as np
import openfhe as fhe

from . import BATCH, rotation_keygen, slot_sum

F = fhe.PKESchemeFeature
DEPTH = {None: 1, 1: 2, 3: 4, 5: 4}   # polynomial degree -> minimum depth
SCALE_BITS = 50
FIRST_MOD_BITS = 60


def _pad(v) -> np.ndarray:
    out = np.zeros(BATCH)
    out[: len(v)] = v
    return out


def make_context(depth: int):
    """CLIENT: CKKS context at HEStd_128_classic; ring dimension chosen by OpenFHE."""
    p = fhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(SCALE_BITS)
    p.SetFirstModSize(FIRST_MOD_BITS)
    p.SetBatchSize(BATCH)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(p)
    for f in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE, F.ADVANCEDSHE):
        cc.Enable(f)
    return cc


def keygen(cc, degree: int | None, sum_method: str = "evalsum"):
    """CLIENT: key pair plus the evaluation keys the circuit uses.

    Slot-summation keys are always needed; a relinearization key only when the
    polynomial multiplies ciphertexts together (degree >= 2).
    """
    keys = cc.KeyGen()
    rotation_keygen(cc, keys.secretKey, sum_method)
    if degree is not None and degree >= 2:
        cc.EvalMultKeyGen(keys.secretKey)
    return keys


def encrypt(cc, public_key, x: np.ndarray):
    """CLIENT: encrypt one preprocessed 11-feature record."""
    return cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext(_pad(x)))


def encode_model(cc, w: np.ndarray, b: float):
    """SERVER: plaintext encodings of the weights and bias."""
    return cc.MakeCKKSPackedPlaintext(_pad(w)), cc.MakeCKKSPackedPlaintext(np.full(BATCH, b))


def evaluate(cc, ct, pt_w, pt_b, coeffs: list[float] | None, sum_method: str = "evalsum") -> list:
    """SERVER: return [ct_z] or [ct_z, ct_p]. No secret key involved."""
    ct_z = cc.EvalAdd(slot_sum(cc, cc.EvalMult(ct, pt_w), sum_method), pt_b)
    if coeffs is None:
        return [ct_z]
    return [ct_z, cc.EvalPoly(ct_z, [float(c) for c in coeffs])]


def decrypt(cc, secret_key, cts: list) -> list[float]:
    """CLIENT: decrypt each result ciphertext and read slot 0."""
    out = []
    for ct in cts:
        pt = cc.Decrypt(ct, secret_key)
        pt.SetLength(1)
        out.append(pt.GetRealPackedValue()[0])
    return out
