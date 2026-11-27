"""BFV encrypted logistic-regression inference with integer quantization.

The feature, weight and bias scales mirror :mod:`fhe_compare.bgv` so that the
two exact-integer schemes can run the same linear and degree-1 circuits.
Results are exact modulo the batching plaintext modulus; input quantization
still introduces error relative to real-valued inference.
"""

from dataclasses import dataclass

import numpy as np
import openfhe as fhe

from . import BATCH, rotation_keygen, slot_sum
from .bgv import batching_prime

F = fhe.PKESchemeFeature
DEPTH = {None: 1, 1: 2}
T_BITS = 31
SX_BITS, SW_BITS = 8, 10


@dataclass(frozen=True)
class Quantizer:
    t: int
    sx: int
    sw: int
    sc: int
    w_q: tuple[int, ...]
    b_q: int
    a0: int
    a1: int
    zmax: int
    pmax: int

    @property
    def s(self) -> int:
        return self.sx * self.sw

    @property
    def half(self) -> int:
        return (self.t - 1) // 2

    @classmethod
    def build(cls, w: np.ndarray, b: float, clip: float, deg1: list[float],
              t_bits: int = T_BITS, sx_bits: int = SX_BITS,
              sw_bits: int = SW_BITS) -> "Quantizer":
        t = batching_prime(t_bits)
        sx, sw = 2**sx_bits, 2**sw_bits
        s = sx * sw
        w_q = tuple(round(v * sw) for v in w)
        b_q = round(b * s)
        zmax = round(clip * sx) * sum(abs(v) for v in w_q) + abs(b_q)
        half = (t - 1) // 2
        c0, c1 = deg1
        fits = [k for k in range(1, 40)
                if abs(round(c1 * 2**k)) * zmax + abs(round(c0 * 2**k)) * s <= half]
        if zmax > half or not fits:
            raise ValueError("plaintext modulus too small for the circuit's worst-case magnitude")
        sc = 2 ** max(fits)
        a0, a1 = round(c0 * sc), round(c1 * sc)
        return cls(t, sx, sw, sc, w_q, b_q, a0, a1, zmax,
                   abs(a1) * zmax + abs(a0) * s)

    def quantize_x(self, x: np.ndarray) -> list[int]:
        """CLIENT: integer features."""
        return [round(v * self.sx) for v in x]

    def exact_z(self, xq: list[int]) -> int:
        return sum(a * c for a, c in zip(xq, self.w_q)) + self.b_q

    def exact_p(self, zq: int) -> int:
        return self.a1 * zq + self.a0 * self.s

    def dequant_z(self, zq: int) -> float:
        return zq / self.s

    def dequant_p(self, pq: int) -> float:
        return pq / (self.sc * self.s)


def _pad(values: list[int]) -> list[int]:
    return list(values) + [0] * (BATCH - len(values))


def make_context(t: int, depth: int):
    """CLIENT: create a batched BFV context at HEStd_128_classic."""
    p = fhe.CCParamsBFVRNS()
    p.SetPlaintextModulus(t)
    p.SetMultiplicativeDepth(depth)
    p.SetBatchSize(BATCH)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(p)
    for feature in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE, F.ADVANCEDSHE):
        cc.Enable(feature)
    if cc.GetPlaintextModulus() != t:
        raise RuntimeError("OpenFHE changed the plaintext modulus")
    return cc


def keygen(cc, degree: int | None, sum_method: str = "evalsum"):
    """CLIENT: generate a key pair and the slot summation keys."""
    keys = cc.KeyGen()
    rotation_keygen(cc, keys.secretKey, sum_method)
    return keys


def encrypt(cc, public_key, xq: list[int]):
    """CLIENT: encrypt one quantized record."""
    return cc.Encrypt(public_key, cc.MakePackedPlaintext(_pad(xq)))


def encode_model(cc, q: Quantizer):
    """SERVER: encode integer model coefficients as packed plaintexts."""
    return {
        "w": cc.MakePackedPlaintext(_pad(list(q.w_q))),
        "b": cc.MakePackedPlaintext([q.b_q] * BATCH),
        "a1": cc.MakePackedPlaintext([q.a1] * BATCH),
        "a0S": cc.MakePackedPlaintext([q.a0 * q.s] * BATCH),
    }


def evaluate(cc, ct, pts: dict, degree: int | None,
             sum_method: str = "evalsum") -> list:
    """SERVER: evaluate the linear score and optionally its degree-1 polynomial."""
    ct_z = cc.EvalAdd(slot_sum(cc, cc.EvalMult(ct, pts["w"]), sum_method), pts["b"])
    if degree is None:
        return [ct_z]
    if degree != 1:
        raise ValueError("only linear and degree-1 circuits are supported")
    return [ct_z, cc.EvalAdd(cc.EvalMult(ct_z, pts["a1"]), pts["a0S"])]


def decrypt(cc, secret_key, cts: list) -> list[int]:
    """CLIENT: decrypt each output and return slot 0's centered residue."""
    results = []
    for ct in cts:
        pt = cc.Decrypt(ct, secret_key)
        pt.SetLength(1)
        results.append(pt.GetPackedValue()[0])
    return results
