"""Phase B: plaintext reference model for the CKKS vs BGV comparison.

Trains logistic regression on the Heart Failure Clinical Records dataset,
fits odd-polynomial sigmoid approximations over the observed score range,
and saves the artifacts that both encrypted pipelines must reuse:

    artifacts/hf_plain.npz   processed arrays, labels, weights, bias, scores
    artifacts/hf_meta.json   preprocessing, model, polynomial, metric metadata

No raw patient rows are printed.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "Heart_failure_clinical_records_dataset.csv"
OUT = ROOT / "artifacts"

SEED = 42
TEST_SIZE = 0.2
TARGET = "DEATH_EVENT"
EXCLUDED = ["time"]  # follow-up duration: not available at prediction time
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
]
LOG1P = ["creatinine_phosphokinase", "serum_creatinine"]
CLIP = 4.0
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]
POLY_DEGREES = [1, 3, 5]
FIT_MARGIN = 1.1  # polynomial fit range R = FIT_MARGIN * max |z_train|
GRID_POINTS = 20001


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "f1": f1_score(y, pred),
        "recall": recall_score(y, pred),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def fmt(m: dict) -> str:
    return f"acc={m['accuracy']:.3f} auc={m['roc_auc']:.3f} f1={m['f1']:.3f} recall={m['recall']:.3f}"


def fit_odd_poly(degree: int, r: float) -> np.ndarray:
    """Least-squares fit p(z) = 0.5 + sum_{odd k<=degree} c_k z^k to sigmoid on [-r, r].

    Returns full coefficients in ascending power order [c0, c1, ..., c_degree].
    sigmoid(z) - 0.5 is odd, so even coefficients above c0 are exactly zero.
    """
    grid = np.linspace(-r, r, GRID_POINTS)
    odd = list(range(1, degree + 1, 2))
    A = np.stack([grid**k for k in odd], axis=1)
    c_odd, *_ = np.linalg.lstsq(A, sigmoid(grid) - 0.5, rcond=None)
    coeffs = np.zeros(degree + 1)
    coeffs[0] = 0.5
    coeffs[odd] = c_odd
    return coeffs


def main() -> None:
    df = pd.read_csv(CSV)
    assert list(df.columns) == FEATURES + EXCLUDED + [TARGET], "unexpected column layout"
    print(f"shape: {df.shape} | missing: {int(df.isna().sum().sum())} | positive rate: {df[TARGET].mean():.3f}")
    print(f"features ({len(FEATURES)}): {FEATURES} | excluded: {EXCLUDED}")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # preprocessing: log1p -> standardize (train stats only) -> clip
    log_cols = [FEATURES.index(c) for c in LOG1P]
    X_log = X_raw.copy()
    X_log[:, log_cols] = np.log1p(X_log[:, log_cols])
    mean = X_log[idx_tr].mean(axis=0)
    scale = X_log[idx_tr].std(axis=0)  # population std, same as sklearn StandardScaler
    X = np.clip((X_log - mean) / scale, -CLIP, CLIP)
    X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    n_clipped = int((np.abs((X_log - mean) / scale) > CLIP).sum())
    print(f"train/test: {len(idx_tr)}/{len(idx_te)} | values clipped to +/-{CLIP}: {n_clipped}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=10000),
        {"C": C_GRID}, scoring="roc_auc", cv=cv,
    ).fit(X_tr, y_tr)
    model = search.best_estimator_
    w, b = model.coef_.ravel(), float(model.intercept_[0])
    assert w.shape == (len(FEATURES),)
    print(f"best C: {search.best_params_['C']} | CV AUC: {search.best_score_:.3f}")

    # recompute scores from saved weights, check against sklearn
    z_tr, z_te = X_tr @ w + b, X_te @ w + b
    assert np.allclose(sigmoid(z_te), model.predict_proba(X_te)[:, 1])
    pred_te = (z_te > 0).astype(int)
    lr_m = metrics(y_te, sigmoid(z_te), pred_te)
    print(f"logistic regression    {fmt(lr_m)}")
    print(f"confusion matrix [[TN FP],[FN TP]]: {lr_m['confusion_matrix']}")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
    rf_m = metrics(y_te, rf.predict_proba(X_te)[:, 1], rf.predict(X_te))
    print(f"random forest (ref)    {fmt(rf_m)}")

    r = FIT_MARGIN * float(np.abs(z_tr).max())
    worst = CLIP * float(np.abs(w).sum()) + abs(b)
    n_out = int((np.abs(z_te) > r).sum())
    print(f"z range train [{z_tr.min():.2f}, {z_tr.max():.2f}] | test [{z_te.min():.2f}, {z_te.max():.2f}]")
    print(f"fit range +/-{r:.2f} | test scores outside: {n_out} | worst-case |z| under clipping: {worst:.2f}")

    polys = {}
    grid = np.linspace(-r, r, GRID_POINTS)
    for d in POLY_DEGREES:
        c = fit_odd_poly(d, r)
        p_te = np.polynomial.polynomial.polyval(z_te, c)
        p_pred = (p_te > 0.5).astype(int)
        max_err = float(np.abs(np.polynomial.polynomial.polyval(grid, c) - sigmoid(grid)).max())
        agree = float((p_pred == pred_te).mean())
        m = metrics(y_te, p_te, p_pred)
        polys[str(d)] = {"coefficients_ascending": c.tolist(), "max_abs_err_on_fit_range": max_err,
                         "decision_agreement_with_sigmoid": agree, "test_metrics": m}
        print(f"poly degree {d}          {fmt(m)}")
        print(f"   max err on fit range={max_err:.4f} | agreement with true sigmoid={agree:.3f}")

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "hf_plain.npz", X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
             idx_train=idx_tr, idx_test=idx_te, w=w, b=np.array(b), z_train=z_tr, z_test=z_te)
    meta = {
        "dataset": {"file": CSV.name, "sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
                    "rows": len(df), "target": TARGET, "excluded_columns": EXCLUDED},
        "split": {"seed": SEED, "test_size": TEST_SIZE, "stratified": True,
                  "n_train": len(idx_tr), "n_test": len(idx_te)},
        "preprocessing": {"feature_order": FEATURES, "log1p_features": LOG1P,
                          "standardize_mean": mean.tolist(), "standardize_scale": scale.tolist(),
                          "fitted_on": "train", "clip": [-CLIP, CLIP],
                          "order": "log1p -> standardize -> clip"},
        "model": {"type": "LogisticRegression", "class_weight": "balanced", "C": search.best_params_["C"],
                  "C_grid": C_GRID, "cv": "StratifiedKFold(5, shuffle=True, seed=42), scoring=roc_auc",
                  "cv_auc": search.best_score_, "weights": w.tolist(), "bias": b,
                  "score": "z = X @ w + b", "decision": "z > 0"},
        "metrics_test": {"logistic_regression": lr_m, "random_forest_reference": rf_m},
        "scores": {"z_train_range": [float(z_tr.min()), float(z_tr.max())],
                   "z_test_range": [float(z_te.min()), float(z_te.max())],
                   "worst_case_abs_z_under_clipping": worst},
        "polynomial": {"form": "p(z) = sum_k c_k z^k, odd k plus c0 = 0.5; least squares vs sigmoid",
                       "fit_range": [-r, r], "fit_margin": FIT_MARGIN, "grid_points": GRID_POINTS,
                       "test_scores_outside_fit_range": n_out, "degrees": polys},
        "note": "Polynomial outputs are not calibrated clinical probabilities.",
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn"]},
    }
    (OUT / "hf_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {OUT / 'hf_plain.npz'} and {OUT / 'hf_meta.json'}")


if __name__ == "__main__":
    main()
