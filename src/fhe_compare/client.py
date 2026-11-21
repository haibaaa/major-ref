"""CLIENT role: owns the patient records and the secret key.

setup    generate context and keys; keep the secret key in client_private/,
         send only the context and evaluation keys to to_server/
encrypt  preprocess raw records with the published parameters, encrypt them
decrypt  decrypt the server's responses, dequantize (BGV), decide from sign(z)
"""

import json
import time

import numpy as np

from . import bgv, ckks
from .protocol import (Layout, Protocol, dump, dump_eval_keys, load_ciphertext, load_context,
                       load_public_key, load_secret_key)


def preprocess(raw: dict[str, float], proto: Protocol) -> np.ndarray:
    """Apply the agreed preprocessing (log1p -> standardize -> clip) to one record."""
    x = np.array([float(raw[f]) for f in proto.feature_order])
    for f in proto.log1p_features:
        i = proto.feature_order.index(f)
        x[i] = np.log1p(x[i])
    x = (x - np.array(proto.standardize_mean)) / np.array(proto.standardize_scale)
    return np.clip(x, -proto.clip, proto.clip)


def _lib(proto: Protocol):
    return ckks if proto.scheme == "ckks" else bgv


def setup(layout: Layout) -> dict:
    proto = Protocol.load(layout.protocol)
    t0 = time.perf_counter()
    cc = ckks.make_context(proto.depth) if proto.scheme == "ckks" else bgv.make_context(proto.t, proto.depth)
    keys = _lib(proto).keygen(cc, proto.degree, proto.sum_method)
    t_keygen = time.perf_counter() - t0

    layout.client_private.mkdir(parents=True, exist_ok=True)
    layout.client_private.chmod(0o700)
    sk_path = layout.client_private / "secret_key.bin"
    sk_path.touch(mode=0o600)
    sk_path.write_bytes(dump(keys.secretKey))
    (layout.client_private / "public_key.bin").write_bytes(dump(keys.publicKey))  # kept by the client
    (layout.client_private / "crypto_context.bin").write_bytes(dump(cc))

    layout.to_server.mkdir(parents=True, exist_ok=True)
    (layout.to_server / "crypto_context.bin").write_bytes(dump(cc))
    relin = proto.scheme == "ckks" and (proto.degree or 0) >= 2
    for name, data in dump_eval_keys(relin).items():
        (layout.to_server / name).write_bytes(data)
    return {"context_and_keygen_s": t_keygen, "ring_dim": cc.GetRingDimension()}


def _load_client(layout: Layout):
    proto = Protocol.load(layout.protocol)
    cc = load_context((layout.client_private / "crypto_context.bin").read_bytes())
    return proto, cc


def encrypt(layout: Layout, records: list[dict[str, float]]) -> dict:
    """Encrypt records; request files are named by sequence number only."""
    proto, cc = _load_client(layout)
    pk = load_public_key((layout.client_private / "public_key.bin").read_bytes())
    layout.requests.mkdir(parents=True, exist_ok=True)
    t = []
    for i, raw in enumerate(records):
        x = preprocess(raw, proto)
        a = time.perf_counter()
        if proto.scheme == "ckks":
            ct = ckks.encrypt(cc, pk, x)
        else:
            ct = bgv.encrypt(cc, pk, [round(v * proto.sx) for v in x])
        t.append(time.perf_counter() - a)
        (layout.requests / f"{i:04d}.ct").write_bytes(dump(ct))
    return {"records": len(records), "encrypt_median_ms": 1e3 * float(np.median(t))}


def decrypt(layout: Layout) -> list[dict]:
    """Decrypt each response. Decision = z > 0; p is a score only within the fit range."""
    proto, cc = _load_client(layout)
    sk = load_secret_key((layout.client_private / "secret_key.bin").read_bytes())
    lib = _lib(proto)
    results = []
    for req in sorted(layout.requests.glob("*.ct")):
        names = ["z"] + (["p"] if proto.degree else [])
        cts = [load_ciphertext((layout.from_server / f"{req.stem}_{n}.ct").read_bytes()) for n in names]
        vals = lib.decrypt(cc, sk, cts)
        if proto.scheme == "bgv":
            raw_ints = list(vals)
            vals = [vals[0] / proto.s] + ([vals[1] / (proto.sc * proto.s)] if proto.degree else [])
        r = {"request": req.stem, "z": float(vals[0]), "decision": int(vals[0] > 0)}
        if proto.degree:
            r["p"] = float(vals[1])
        if proto.scheme == "bgv":
            r["raw_residues"] = raw_ints
        results.append(r)
    (layout.client_private / "decrypted.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
