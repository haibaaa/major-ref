"""Public protocol parameters, file layout, and OpenFHE (de)serialization helpers.

Directory layout of one demo run (all under demo/<run>/, git-ignored):
    public/protocol.json     agreed parameters, published by the model owner (server)
    client_private/          secret key and client-side bookkeeping; never sent
    to_server/               context, evaluation keys, request ciphertexts
    from_server/             response ciphertexts
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import openfhe as fhe


@dataclass(frozen=True)
class Protocol:
    """Parameters both parties agree on. Contains no patient data and no keys."""
    scheme: str                  # "ckks" | "bgv"
    degree: int | None           # sigmoid polynomial degree, None = linear score only
    depth: int
    batch: int
    sum_method: str
    feature_order: list[str]
    log1p_features: list[str]
    standardize_mean: list[float]
    standardize_scale: list[float]
    clip: float
    # BGV only (None for CKKS)
    t: int | None = None
    sx: int | None = None
    s: int | None = None         # Sx * Sw, scale of z_q
    sc: int | None = None        # coefficient scale of P_q (scale Sc * S)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Protocol":
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def protocol(self) -> Path: return self.root / "public" / "protocol.json"
    @property
    def client_private(self) -> Path: return self.root / "client_private"
    @property
    def to_server(self) -> Path: return self.root / "to_server"
    @property
    def requests(self) -> Path: return self.to_server / "requests"
    @property
    def from_server(self) -> Path: return self.root / "from_server"


# ---- serialization (OpenFHE BINARY) ---------------------------------------

def dump(obj) -> bytes:
    """Serialize a CryptoContext, key or Ciphertext."""
    return fhe.Serialize(obj, fhe.BINARY)


def load_context(data: bytes):
    return fhe.DeserializeCryptoContextString(data, fhe.BINARY)


def load_ciphertext(data: bytes):
    return fhe.DeserializeCiphertextString(data, fhe.BINARY)


def load_secret_key(data: bytes):
    return fhe.DeserializePrivateKeyString(data, fhe.BINARY)


def load_public_key(data: bytes):
    return fhe.DeserializePublicKeyString(data, fhe.BINARY)


def dump_eval_keys(relin: bool) -> dict[str, bytes]:
    """Serialize the evaluation keys currently in OpenFHE's key cache."""
    out = {"rotation_keys.bin": fhe.SerializeEvalAutomorphismKeyString(fhe.BINARY)}
    if relin:
        out["relin_key.bin"] = fhe.SerializeEvalMultKeyString(fhe.BINARY)
    return out


def load_eval_keys(directory: Path) -> list[str]:
    """Insert serialized evaluation keys from `directory` into the key cache."""
    loaded = []
    rot = directory / "rotation_keys.bin"
    fhe.DeserializeEvalAutomorphismKeyString(rot.read_bytes(), fhe.BINARY)
    loaded.append(rot.name)
    relin = directory / "relin_key.bin"
    if relin.exists():
        fhe.DeserializeEvalMultKeyString(relin.read_bytes(), fhe.BINARY)
        loaded.append(relin.name)
    return loaded

