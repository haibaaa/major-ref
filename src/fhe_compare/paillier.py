"""Paillier encrypted linear-score baseline.

Paillier is additively homomorphic: encrypted feature values can be multiplied
by plaintext model weights and summed. It does not support ciphertext-by-
ciphertext multiplication, so this module intentionally supports only the
linear logit and an optional affine (degree-1) score transform.
"""

from phe import paillier


def keygen(n_length: int = 2048):
    """CLIENT: generate a Paillier public/private key pair."""
    return paillier.generate_paillier_keypair(n_length=n_length)


def encrypt(public_key, features: list[float]):
    """CLIENT: encrypt each feature as an individual Paillier ciphertext."""
    return [public_key.encrypt(float(value)) for value in features]


def evaluate(encrypted_features, weights: list[float], bias: float,
             degree1: list[float] | None = None):
    """SERVER: return an encrypted linear score and optional affine score.

    The model coefficients remain plaintext at the server. ``degree1`` uses
    ascending polynomial coefficients [c0, c1], which Paillier can apply
    because this polynomial is affine. Higher-degree polynomials are rejected.
    """
    if len(encrypted_features) != len(weights):
        raise ValueError("feature and weight counts differ")
    z = sum((ct * float(weight) for ct, weight in zip(encrypted_features, weights)),
            encrypted_features[0] * 0)
    z += float(bias)
    outputs = [z]
    if degree1 is not None:
        if len(degree1) != 2:
            raise ValueError("Paillier supports only an affine polynomial [c0, c1]")
        c0, c1 = degree1
        outputs.append(z * float(c1) + float(c0))
    return outputs


def decrypt(private_key, encrypted_outputs) -> list[float]:
    """CLIENT: decrypt score ciphertexts."""
    return [float(private_key.decrypt(value)) for value in encrypted_outputs]


def public_key_bytes(public_key) -> int:
    """Approximate public-key wire size, as the decimal encoding of ``n``."""
    return len(str(public_key.n).encode("ascii"))


def ciphertext_bytes(encrypted_value) -> int:
    """Approximate ciphertext wire size, including its fixed-point exponent."""
    return len(str(encrypted_value.ciphertext(be_secure=False)).encode("ascii")) + 8
