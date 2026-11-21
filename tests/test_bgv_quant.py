import numpy as np

from fhe_compare.bgv import Quantizer, batching_prime, is_prime


def test_batching_prime():
    t = batching_prime(31)
    assert is_prime(t) and (t - 1) % (1 << 16) == 0 and t.bit_length() == 31


def test_is_prime_small():
    assert [n for n in range(30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_bounds_cover_boundary_records(model):
    q = Quantizer.build(model.w, model.b, model.clip, model.poly[1])
    assert q.zmax <= q.half and q.pmax <= q.half
    for x in model.edge_records():
        zq = q.exact_z(q.quantize_x(x))
        assert abs(zq) <= q.zmax and abs(q.exact_p(zq)) <= q.pmax
    # the exact bound is attained by one of the two corner records
    assert max(abs(q.exact_z(q.quantize_x(x))) for x in model.edge_records()) == q.zmax


def test_quantization_preserves_test_decisions(model):
    q = Quantizer.build(model.w, model.b, model.clip, model.poly[1])
    z = np.array([q.dequant_z(q.exact_z(q.quantize_x(x))) for x in model.X_test])
    assert np.abs(z - model.z_test).max() < np.abs(model.z_test).min()
    assert ((z > 0) == (model.z_test > 0)).all()
