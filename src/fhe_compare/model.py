"""Load the plaintext artifacts produced by scripts/train_plain.py."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Model:
    X_test: np.ndarray        # preprocessed, clipped features, shape (n, 11)
    y_test: np.ndarray
    w: np.ndarray             # logistic-regression weights, shape (11,)
    b: float
    z_test: np.ndarray        # plaintext scores X_test @ w + b
    clip: float               # features lie in [-clip, clip]
    fit_range: float          # polynomial fitted on [-fit_range, fit_range]
    poly: dict[int, list[float]]  # degree -> coefficients, ascending powers

    def edge_records(self) -> np.ndarray:
        """Two synthetic records at the clip box corners aligned with +/-sign(w)."""
        s = np.sign(self.w)
        return np.stack([self.clip * s, -self.clip * s])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(root: Path = ROOT) -> Model:
    art = np.load(root / "artifacts" / "hf_plain.npz")
    meta = json.loads((root / "artifacts" / "hf_meta.json").read_text())
    w, b = art["w"], float(art["b"])
    if not (np.allclose(w, meta["model"]["weights"]) and np.isclose(b, meta["model"]["bias"])):
        raise ValueError("hf_plain.npz and hf_meta.json describe different models")
    if not np.allclose(art["X_test"] @ w + b, art["z_test"]):
        raise ValueError("saved z_test is inconsistent with saved weights")
    return Model(
        X_test=art["X_test"], y_test=art["y_test"], w=w, b=b, z_test=art["z_test"],
        clip=float(meta["preprocessing"]["clip"][1]),
        fit_range=float(meta["polynomial"]["fit_range"][1]),
        poly={int(d): v["coefficients_ascending"] for d, v in meta["polynomial"]["degrees"].items()},
    )
