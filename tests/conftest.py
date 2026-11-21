import pytest

from fhe_compare.model import ROOT, load_model


@pytest.fixture(scope="session")
def model():
    if not (ROOT / "artifacts" / "hf_plain.npz").exists():
        pytest.skip("artifacts missing: run `uv run python scripts/train_plain.py` first")
    return load_model()
