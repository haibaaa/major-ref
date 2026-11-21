"""Serialization round trips and client-side preprocessing."""

import numpy as np
import openfhe as fhe
import pandas as pd
import pytest

from fhe_compare import bgv, ckks, client, server
from fhe_compare.model import ROOT
from fhe_compare.protocol import (Layout, Protocol, dump, dump_eval_keys, load_ciphertext, load_context,
                                  load_eval_keys, load_secret_key)

CSV = ROOT / "data" / "Heart_failure_clinical_records_dataset.csv"


@pytest.mark.parametrize("scheme,degree", [("ckks", 1), ("ckks", 3), ("bgv", 1)])
def test_roundtrip_after_clearing_library_state(model, tmp_path, scheme, degree):
    """Serialize everything, clear OpenFHE's global caches, reload, evaluate, decrypt."""
    x = model.X_test[0]
    if scheme == "ckks":
        cc = ckks.make_context(ckks.DEPTH[degree])
        keys = ckks.keygen(cc, degree)
        payload = x
    else:
        q = bgv.Quantizer.build(model.w, model.b, model.clip, model.poly[1])
        cc = bgv.make_context(q.t, bgv.DEPTH[degree])
        keys = bgv.keygen(cc, degree)
        payload = q.quantize_x(x)
    lib = ckks if scheme == "ckks" else bgv
    blob_cc, blob_sk = dump(cc), dump(keys.secretKey)
    blob_ct = dump(lib.encrypt(cc, keys.publicKey, payload))
    for name, data in dump_eval_keys(relin=scheme == "ckks" and degree >= 2).items():
        (tmp_path / name).write_bytes(data)
    del cc, keys
    fhe.CryptoContext.ClearStaticMapsAndVectors()

    cc2 = load_context(blob_cc)
    load_eval_keys(tmp_path)
    ct = load_ciphertext(blob_ct)
    if scheme == "ckks":
        outs = ckks.evaluate(cc2, ct, *ckks.encode_model(cc2, model.w, model.b), model.poly[degree])
    else:
        outs = bgv.evaluate(cc2, ct, bgv.encode_model(cc2, q), degree)
    # round-trip the outputs too, then decrypt with the reloaded secret key
    outs = [load_ciphertext(dump(o)) for o in outs]
    vals = lib.decrypt(cc2, load_secret_key(blob_sk), outs)
    z = x @ model.w + model.b
    if scheme == "ckks":
        assert abs(vals[0] - z) < 1e-6
        assert abs(vals[1] - np.polynomial.polynomial.polyval(z, model.poly[degree])) < 1e-6
    else:
        zq = q.exact_z(payload)
        assert vals == [zq, q.exact_p(zq)]


def test_client_preprocessing_matches_training_pipeline(model, tmp_path):
    layout = Layout(tmp_path)
    proto = server.init(layout, "ckks", 1)
    assert Protocol.load(layout.protocol) == proto
    idx = np.load(ROOT / "artifacts" / "hf_plain.npz")["idx_test"]
    rows = pd.read_csv(CSV).iloc[idx].to_dict("records")
    X = np.stack([client.preprocess(r, proto) for r in rows])
    np.testing.assert_allclose(X, model.X_test, rtol=0, atol=1e-12)
