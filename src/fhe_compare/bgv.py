"""BGV encrypted logistic-regression inference with integer quantization.

Quantization:  x_q = round(x*Sx) (CLIENT), w_q = round(w*Sw), b_q = round(b*Sx*Sw),
degree-1 coefficients a_k = round(c_k*Sc) (SERVER-side model encoding).
Circuit (SERVER): z_q = slot_sum(x_q * w_q) + b_q;  optionally P_q = a1*z_q + a0*S.
All values are residues mod t in the centered range; `Quantizer` checks the
exact worst-case magnitudes over the clip box against (t-1)/2 so no
wraparound can occur for any clipped input.

Only the linear and degree-1 circuits fit one ~31-bit plaintext modulus
(see scripts/bgv_quant.py and scripts/bgv_modulus_probe.py).
"""

from dataclasses import dataclass

import numpy as np
import openfhe as fhe

from . import BATCH, rotation_keygen, slot_sum

F = fhe.PKESchemeFeature
DEPTH = {None: 1, 1: 2}   # polynomial degree -> depth used
T_BITS = 31
SX_BITS, SW_BITS = 8, 10


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
    """Smallest prime >= 2^(bits-1) with t = 1 (mod m); m = 2^16 allows N <= 2^15."""
    k = (1 << (bits - 1)) // m + 1
    while not is_prime(k * m + 1):
        k += 1
    return k * m + 1


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
    zmax: int   # exact max |z_q| over the clip box
    pmax: int   # max |P_q| bound (triangle inequality)

    @property
    def s(self) -> int:
        return self.sx * self.sw

    @property
    def half(self) -> int:
        return (self.t - 1) // 2

    @classmethod
    def build(cls, w: np.ndarray, b: float, clip: float, deg1: list[float],
              t_bits: int = T_BITS, sx_bits: int = SX_BITS, sw_bits: int = SW_BITS) -> "Quantizer":
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
        return cls(t, sx, sw, sc, w_q, b_q, a0, a1, zmax, abs(a1) * zmax + abs(a0) * s)

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


def _pad(v: list[int]) -> list[int]:
    return list(v) + [0] * (BATCH - len(v))


def make_context(t: int, depth: int):
    """CLIENT: BGV context at HEStd_128_classic (default FLEXIBLEAUTO scaling)."""
    p = fhe.CCParamsBGVRNS()
    p.SetPlaintextModulus(t)
    p.SetMultiplicativeDepth(depth)
    p.SetBatchSize(BATCH)
    p.SetSecurityLevel(fhe.SecurityLevel.HEStd_128_classic)
    cc = fhe.GenCryptoContext(p)
    for f in (F.PKE, F.KEYSWITCH, F.LEVELEDSHE, F.ADVANCEDSHE):
        cc.Enable(f)
    if cc.GetPlaintextModulus() != t:
        raise RuntimeError("OpenFHE changed the plaintext modulus")
    return cc


def keygen(cc, degree: int | None, sum_method: str = "evalsum"):
    """CLIENT: key pair plus slot-summation keys. The circuit has only ciphertext-plaintext
    products, so no relinearization key is generated."""
    keys = cc.KeyGen()
    rotation_keygen(cc, keys.secretKey, sum_method)
    return keys


def encrypt(cc, public_key, xq: list[int]):
    """CLIENT: encrypt one quantized record."""
    return cc.Encrypt(public_key, cc.MakePackedPlaintext(_pad(xq)))


def encode_model(cc, q: Quantizer):
    """SERVER: plaintext encodings of quantized weights, bias and degree-1 coefficients."""
    return {
        "w": cc.MakePackedPlaintext(_pad(list(q.w_q))),
        "b": cc.MakePackedPlaintext([q.b_q] * BATCH),
        "a1": cc.MakePackedPlaintext([q.a1] * BATCH),
        "a0S": cc.MakePackedPlaintext([q.a0 * q.s] * BATCH),
    }


def evaluate(cc, ct, pts: dict, degree: int | None, sum_method: str = "evalsum") -> list:
    """SERVER: return [ct_z] or [ct_z, ct_p]. No secret key involved.

    Only slot 0 of the result is meaningful.
    """
    ct_z = cc.EvalAdd(slot_sum(cc, cc.EvalMult(ct, pts["w"]), sum_method), pts["b"])
    if degree is None:
        return [ct_z]
    if degree != 1:
        raise ValueError("only linear and degree-1 circuits fit a single plaintext modulus")
    return [ct_z, cc.EvalAdd(cc.EvalMult(ct_z, pts["a1"]), pts["a0S"])]


def decrypt(cc, secret_key, cts: list) -> list[int]:
    """CLIENT: decrypt each result ciphertext and read slot 0 (centered residue)."""
    out = []
    for ct in cts:
        pt = cc.Decrypt(ct, secret_key)
        pt.SetLength(1)
        out.append(pt.GetPackedValue()[0])
    return out
