"""Serialized sizes (bytes, OpenFHE BINARY format) of crypto objects."""

import tempfile
from pathlib import Path

import openfhe as fhe


def nbytes(obj) -> int:
    """Size of a CryptoContext, PublicKey, PrivateKey or Ciphertext."""
    return len(fhe.Serialize(obj, fhe.BINARY))


def eval_key_sizes(cc, relin_generated: bool) -> dict[str, int]:
    """Sizes of the relinearization key and rotation (automorphism) keys held by `cc`.

    OpenFHE serializes an empty key map as a few bytes, so a key set that was
    not generated is reported as 0 rather than its empty-map size.
    """
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for name, fn, generated in [("relin_key", cc.SerializeEvalMultKey, relin_generated),
                                    ("rotation_keys", cc.SerializeEvalAutomorphismKey, True)]:
            p = Path(d) / name
            ok = generated and fn(str(p), fhe.BINARY)
            out[name] = p.stat().st_size if ok and p.exists() else 0
    return out
