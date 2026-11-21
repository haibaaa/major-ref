import numpy as np

from fhe_compare.model import ROOT


def test_artifacts_consistent(model):
    assert model.w.shape == (11,)
    assert model.X_test.shape == (60, 11)
    assert np.abs(model.X_test).max() <= model.clip
    np.testing.assert_allclose(model.X_test @ model.w + model.b, model.z_test)


def test_split_disjoint():
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    assert not set(art["idx_train"]) & set(art["idx_test"])
    assert len(art["idx_train"]) + len(art["idx_test"]) == 299


def test_polynomials_odd_around_half(model):
    for d, c in model.poly.items():
        assert len(c) == d + 1 and c[0] == 0.5
        assert all(c[k] == 0 for k in range(2, d + 1, 2))
