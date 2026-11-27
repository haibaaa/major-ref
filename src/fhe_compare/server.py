"""SERVER role: model owner and evaluator.

Holds the model (weights from hf_meta.json, which contains no patient rows),
publishes the protocol, and evaluates request ciphertexts. It loads only the
crypto context and evaluation keys from to_server/; it never receives the
secret key, the public key, or plaintext features.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import BATCH, bfv, bgv, ckks
from .model import ROOT
from .protocol import Layout, Protocol, dump, load_ciphertext, load_context, load_eval_keys


def _meta(root: Path = ROOT) -> dict:
    return json.loads((root / "artifacts" / "hf_meta.json").read_text())


def _quantizer(meta: dict) -> bgv.Quantizer:
    return bgv.Quantizer.build(np.array(meta["model"]["weights"]), meta["model"]["bias"],
                               meta["preprocessing"]["clip"][1],
                               meta["polynomial"]["degrees"]["1"]["coefficients_ascending"])


def init(layout: Layout, scheme: str, degree: int | None, sum_method: str = "evalsum") -> Protocol:
    """Publish the protocol parameters for this run."""
    meta = _meta()
    pre = meta["preprocessing"]
    common = dict(scheme=scheme, degree=degree, batch=BATCH, sum_method=sum_method,
                  feature_order=pre["feature_order"], log1p_features=pre["log1p_features"],
                  standardize_mean=pre["standardize_mean"], standardize_scale=pre["standardize_scale"],
                  clip=pre["clip"][1])
    if scheme == "ckks":
        proto = Protocol(depth=ckks.DEPTH[degree], **common)
    elif scheme in ("bgv", "bfv"):
        lib = bfv if scheme == "bfv" else bgv
        q = lib.Quantizer.build(np.array(meta["model"]["weights"]), meta["model"]["bias"],
                                pre["clip"][1],
                                meta["polynomial"]["degrees"]["1"]["coefficients_ascending"])
        proto = Protocol(depth=lib.DEPTH[degree], t=q.t, sx=q.sx, s=q.s, sc=q.sc, **common)
    else:
        raise ValueError(scheme)
    proto.save(layout.protocol)
    return proto


def evaluate(layout: Layout) -> dict:
    """Evaluate every request ciphertext; write one response file per output."""
    t0 = time.perf_counter()
    proto = Protocol.load(layout.protocol)
    meta = _meta()
    cc = load_context((layout.to_server / "crypto_context.bin").read_bytes())
    key_files = load_eval_keys(layout.to_server)
    if proto.scheme == "ckks":
        pts = ckks.encode_model(cc, np.array(meta["model"]["weights"]), meta["model"]["bias"])
        coeffs = meta["polynomial"]["degrees"][str(proto.degree)]["coefficients_ascending"] if proto.degree else None
        run = lambda ct: ckks.evaluate(cc, ct, *pts, coeffs, proto.sum_method)
    else:
        lib = bfv if proto.scheme == "bfv" else bgv
        q = lib.Quantizer.build(np.array(meta["model"]["weights"]), meta["model"]["bias"],
                                proto.clip, meta["polynomial"]["degrees"]["1"]["coefficients_ascending"])
        assert (q.t, q.sx, q.s, q.sc) == (proto.t, proto.sx, proto.s, proto.sc), "protocol/model mismatch"
        pts = lib.encode_model(cc, q)
        run = lambda ct: lib.evaluate(cc, ct, pts, proto.degree, proto.sum_method)
    t_load = time.perf_counter() - t0

    layout.from_server.mkdir(parents=True, exist_ok=True)
    t_eval = []
    requests = sorted(layout.requests.glob("*.ct"))
    for req in requests:
        ct = load_ciphertext(req.read_bytes())
        a = time.perf_counter()
        outs = run(ct)
        t_eval.append(time.perf_counter() - a)
        for name, out in zip(("z", "p"), outs):
            (layout.from_server / f"{req.stem}_{name}.ct").write_bytes(dump(out))
    return {"requests": len(requests), "eval_keys_loaded": key_files,
            "load_context_keys_model_s": t_load, "evaluate_median_ms": 1e3 * float(np.median(t_eval))}
