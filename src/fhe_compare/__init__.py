"""CKKS vs BGV encrypted logistic-regression inference (OpenFHE).

Modules:
    model   load and validate the plaintext artifacts shared by both schemes
    ckks    CKKS context, keys, encryption, circuit evaluation, decryption
    bgv     BGV quantization, context, keys, encryption, circuit, decryption
    sizes   serialized-size measurement for contexts, keys and ciphertexts

Functions are labelled CLIENT (holds the secret key) or SERVER (sees only
ciphertexts, the public context, and evaluation keys).
"""

BATCH = 16  # slots per record: power of two >= 11 features
ROTATIONS = (1, 2, 4, 8)  # left rotations whose rotate-and-add puts sum(slots 0..15) in slot 0
SUM_METHODS = ("evalsum", "rotations")


def slot_sum(cc, ct, method: str = "evalsum"):
    """SERVER: put the sum of slots 0..BATCH-1 into slot 0.

    "evalsum":   OpenFHE EvalSum(ct, BATCH). Faster in our measurements; EvalSumKeyGen
                 produced 6 automorphism keys for this setup (observed, OpenFHE 1.6.1).
    "rotations": rotate-and-add with only the 4 rotation keys it uses; smaller key
                 material, slower evaluation (see scripts/benchmark.py --sum rotations).
    Only slot 0 is guaranteed: BGV EvalSum leaves suffix sums in the other slots.
    """
    if method == "evalsum":
        return cc.EvalSum(ct, BATCH)
    if method == "rotations":
        for k in ROTATIONS:
            ct = cc.EvalAdd(ct, cc.EvalRotate(ct, k))
        return ct
    raise ValueError(f"unknown sum method {method!r}")


def rotation_keygen(cc, secret_key, method: str = "evalsum") -> None:
    """CLIENT: generate the rotation keys `slot_sum(method=...)` needs."""
    if method == "evalsum":
        cc.EvalSumKeyGen(secret_key)
    elif method == "rotations":
        cc.EvalRotateKeyGen(secret_key, list(ROTATIONS))
    else:
        raise ValueError(f"unknown sum method {method!r}")
