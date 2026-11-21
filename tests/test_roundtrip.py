"""Encrypt -> evaluate -> decrypt on a few records for each scheme and circuit."""

import numpy as np
import pytest

from fhe_compare import bgv, ckks

N_RECORDS = 3


@pytest.mark.parametrize("sum_method", ["evalsum", "rotations"])
@pytest.mark.parametrize("degree", [None, 1, 3])
def test_ckks(model, degree, sum_method):
    cc = ckks.make_context(ckks.DEPTH[degree])
    keys = ckks.keygen(cc, degree, sum_method)
    pts = ckks.encode_model(cc, model.w, model.b)
    coeffs = model.poly[degree] if degree else None
    xs = np.vstack([model.X_test[:N_RECORDS], model.edge_records()])
    for x in xs:
        out = ckks.decrypt(cc, keys.secretKey, ckks.evaluate(cc, ckks.encrypt(cc, keys.publicKey, x), *pts, coeffs, sum_method))
        z = x @ model.w + model.b
        assert abs(out[0] - z) < 1e-6
        if coeffs:
            assert abs(out[1] - np.polynomial.polynomial.polyval(z, coeffs)) < 1e-6


@pytest.mark.parametrize("sum_method", ["evalsum", "rotations"])
@pytest.mark.parametrize("degree", [None, 1])
def test_bgv_exact(model, degree, sum_method):
    q = bgv.Quantizer.build(model.w, model.b, model.clip, model.poly[1])
    cc = bgv.make_context(q.t, bgv.DEPTH[degree])
    keys = bgv.keygen(cc, degree, sum_method)
    pts = bgv.encode_model(cc, q)
    xs = np.vstack([model.X_test[:N_RECORDS], model.edge_records()])
    for x in xs:
        xq = q.quantize_x(x)
        out = bgv.decrypt(cc, keys.secretKey, bgv.evaluate(cc, bgv.encrypt(cc, keys.publicKey, xq), pts, degree, sum_method))
        assert out[0] == q.exact_z(xq)
        if degree:
            assert out[1] == q.exact_p(q.exact_z(xq))


def test_bgv_rejects_unsupported_degree(model):
    q = bgv.Quantizer.build(model.w, model.b, model.clip, model.poly[1])
    cc = bgv.make_context(q.t, 2)
    keys = bgv.keygen(cc, None)
    with pytest.raises(ValueError):
        bgv.evaluate(cc, bgv.encrypt(cc, keys.publicKey, [0] * 11), bgv.encode_model(cc, q), 3)
