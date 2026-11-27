# Architecture

## 1. Repository layout

```
.
├── CLAUDE.md                 authoritative goals, rules, phase plan
├── pyproject.toml / uv.lock  uv project; Python 3.12; uv_build backend
├── .python-version           3.12
├── .gitignore                ignores .venv, data/*.csv, artifacts/, demo/, keys, *.bin, *.ct, results/*.npz
├── data/                     Heart_failure_clinical_records_dataset.csv   (NOT in git)
├── artifacts/                hf_plain.npz, hf_meta.json                   (NOT in git; regenerate)
├── demo/                     client/server exchange dirs, contain SECRET KEYS (NOT in git)
├── docs/                     this documentation
├── results/                  machine-readable results (committed; small JSON/MD)
├── scripts/                  runnable experiments and CLIs (see §3)
├── src/fhe_compare/          canonical library code (see §2)
└── tests/                    pytest suite (22 tests)
```

## 2. Package `src/fhe_compare`

Every cryptographic function's docstring starts with **CLIENT** (may use the
secret key / plaintext features) or **SERVER** (sees only context, evaluation
keys, ciphertexts and the model). Keep that convention.

| Module | Role | Key contents |
|---|---|---|
| `__init__.py` | shared | `BATCH = 16`; `ROTATIONS = (1, 2, 4, 8)`; `SUM_METHODS`; `slot_sum(cc, ct, method)` (SERVER); `rotation_keygen(cc, sk, method)` (CLIENT) |
| `model.py` | shared | `Model` dataclass (X_test, y_test, w, b, z_test, clip, fit_range, poly); `load_model()` validates npz vs json and `z_test == X_test @ w + b`; `Model.edge_records()`; `sigmoid` |
| `ckks.py` | both | `DEPTH = {None: 1, 1: 2, 3: 4, 5: 4}`, `SCALE_BITS = 50`, `FIRST_MOD_BITS = 60`; `make_context`, `keygen`, `encrypt` (CLIENT); `encode_model`, `evaluate` (SERVER); `decrypt` (CLIENT) |
| `bgv.py` | both | `DEPTH = {None: 1, 1: 2}`, `T_BITS = 31`, `SX_BITS = 8`, `SW_BITS = 10`; `is_prime`, `batching_prime`; `Quantizer` (scales, quantized model, exact bounds, exact reference arithmetic, dequantization); `make_context`, `keygen`, `encrypt`, `encode_model`, `evaluate`, `decrypt` |
| `bfv.py` | both | BFV counterpart to BGV using the same quantization scales and linear/degree-1 circuit; modulus and wraparound behavior still require empirical validation |
| `paillier.py` | both | Additive-encryption baseline using `phe`; encrypts each feature separately and evaluates the linear score with plaintext weights; not packed FHE and does not support higher-degree polynomial evaluation |
| `sizes.py` | measurement | `nbytes(obj)` via `openfhe.Serialize(obj, BINARY)`; `eval_key_sizes(cc, relin_generated)` (reports 0 for key sets not generated) |
| `protocol.py` | shared | `Protocol` dataclass (public agreed parameters, JSON); `Layout` (demo directory paths); `dump`, `load_context`, `load_ciphertext`, `load_secret_key`, `load_public_key`, `dump_eval_keys`, `load_eval_keys` |
| `client.py` | CLIENT | `preprocess(raw, proto)`; `setup`, `encrypt`, `decrypt` (file-based) |
| `server.py` | SERVER | `init` (publishes protocol from `hf_meta.json`), `evaluate` (file-based) |

### Scheme function signatures (current)

```python
# CKKS
cc   = ckks.make_context(depth)
keys = ckks.keygen(cc, degree, sum_method="evalsum")      # degree None|1|3|5
ct   = ckks.encrypt(cc, keys.publicKey, x)                # x: 11 floats (preprocessed)
pt_w, pt_b = ckks.encode_model(cc, w, b)
outs = ckks.evaluate(cc, ct, pt_w, pt_b, coeffs_or_None, sum_method="evalsum")   # [ct_z] or [ct_z, ct_p]
vals = ckks.decrypt(cc, keys.secretKey, outs)             # floats from slot 0

# BGV
q    = bgv.Quantizer.build(w, b, clip, deg1_coeffs)       # t, scales, bounds
cc   = bgv.make_context(q.t, depth)
keys = bgv.keygen(cc, degree, sum_method="evalsum")       # degree None|1
ct   = bgv.encrypt(cc, keys.publicKey, q.quantize_x(x))
pts  = bgv.encode_model(cc, q)                            # dict: w, b, a1, a0S
outs = bgv.evaluate(cc, ct, pts, degree, sum_method="evalsum")   # raises ValueError for degree 3/5
vals = bgv.decrypt(cc, keys.secretKey, outs)              # centered residues (ints) from slot 0
```

## 3. Scripts

| Script | Phase | Uses package? | Purpose |
|---|---|---|---|
| `record_env.py` | A | no | writes `results/environment.json` (Python, uv, OpenFHE, numpy, sklearn, OS, kernel, glibc, CPU, logical CPUs) |
| `train_plain.py` | B | no | plaintext training, polynomial fits, artifacts |
| `ckks_smoke.py`, `bgv_smoke.py` | C | no | minimal API checks; BGV includes a deliberate wraparound (300·300 mod 65537) |
| `ckks_linear.py` | D | no | first encrypted linear score (EvalSum) |
| `ckks_poly.py` | D | no | depth search per degree, boundary records |
| `bgv_quant.py` | E | no | quantization sweep (no encryption) |
| `bgv_modulus_probe.py` | E | no | largest usable plaintext modulus, per scaling technique |
| `bgv_linear.py` | E | imports `bgv_modulus_probe.batching_prime` | first BGV linear + degree-1 inference |
| `benchmark.py` | F | yes | controlled benchmark (subprocess per config) |
| `benchmark_paillier.py` | extension | yes | Paillier additive linear-score benchmark; record count is configurable |
| `margin_stress.py` | extension | yes | controlled synthetic logits near zero; quantization error bounds and decision flips |
| `report.py` | F/H | no | Markdown tables from benchmark JSON |
| `fhe_server.py`, `fhe_client.py` | G | yes | role CLIs |
| `demo_client_server.py` | G | yes | runs roles as separate processes, audits, compares |

Phase D/E scripts are the historical record of how parameters were found; the
package re-implements the final circuits. If you change a circuit, change the
package and the benchmark, not the exploratory scripts.

## 4. Data pipeline (plaintext, `train_plain.py`)

1. Read CSV; assert columns `[11 features] + ['time', 'DEATH_EVENT']`.
2. Feature order (fixed everywhere): `age, anaemia, creatinine_phosphokinase,
   diabetes, ejection_fraction, high_blood_pressure, platelets,
   serum_creatinine, serum_sodium, sex, smoking`. `time` is excluded (follow-up
   duration, not available at prediction time).
3. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` on row
   indices → 239 train / 60 test.
4. `log1p` on `creatinine_phosphokinase` and `serum_creatinine`.
5. Standardize with **training** mean and population std (`np.std`, same as
   `StandardScaler`).
6. Clip to [-4, 4] (7 values clipped in the whole dataset).
7. `GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=10000),
   C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}, scoring="roc_auc",
   cv=StratifiedKFold(5, shuffle=True, random_state=42))`.
8. Score `z = X @ w + b`; decision `z > 0` (equivalent to sigmoid > 0.5).
9. Random forest reference: 300 trees, `class_weight="balanced"`, seed 42
   (reference only; not part of the encrypted comparison).
10. Polynomial fit range `R = 1.1 · max|z_train|` = ±4.3188.
11. Odd least-squares fits `p(z) = 0.5 + Σ_{odd k} c_k z^k` to the sigmoid on a
    20001-point grid over [-R, R], degrees 1, 3, 5.

The client re-implements steps 4–6 in `client.preprocess` using the
statistics published in `protocol.json`; a test asserts it reproduces
`X_test` to 1e-12.

### Current model (from `artifacts/hf_meta.json`)

| feature | weight | | feature | weight |
|---|---|---|---|---|
| age | 0.3347 | | platelets | −0.0834 |
| anaemia | 0.2387 | | serum_creatinine | 0.7073 |
| creatinine_phosphokinase | 0.2035 | | serum_sodium | −0.1484 |
| diabetes | 0.1892 | | sex | −0.0474 |
| ejection_fraction | −0.5430 | | smoking | 0.1311 |
| high_blood_pressure | 0.2463 | | **bias** | −0.1875 |

Score ranges: train [−3.157, 3.926], test [−2.903, 2.592]; worst case under
clipping `4·Σ|w| + |b|` = 11.680. Smallest test |z| = 0.0056 (the margin any
quantization error must stay below to preserve decisions).

Polynomial coefficients (ascending powers):

| degree | c0 | c1 | c3 | c5 | max err on fit range |
|---|---|---|---|---|---|
| 1 | 0.5 | 0.145646 | | | 0.1422 |
| 3 | 0.5 | 0.210824 | −0.00582341 | | 0.0455 |
| 5 | 0.5 | 0.236587 | −0.0122687 | 0.000310967 | 0.0138 |

## 5. Encrypted circuits

### Slot layout (both schemes)

One patient record per ciphertext. Features in slots 0..10, zeros in slots
11..15, batch size 16 (`SetBatchSize(16)`). The weight plaintext also has zeros
in 11..15. Only **slot 0** of a result is meaningful.

### Slot summation (`slot_sum`)

- `"evalsum"` (default): `cc.EvalSum(ct, 16)` with keys from `EvalSumKeyGen`.
  In CKKS every slot ends up holding the total; in **BGV slot i holds the sum
  of slots i..15 (suffix sums)** — slot 0 is still the total.
  `EvalSumKeyGen` produced 6 automorphism keys here.
- `"rotations"`: `ct += Rotate(ct, k)` for k = 1, 2, 4, 8 (positive = left), keys
  from `EvalRotateKeyGen(sk, [1, 2, 4, 8])` (4 automorphism keys). ~1/3 smaller
  rotation-key material, ~50 % slower summation (measured).

### CKKS circuit (`ckks.evaluate`)

```
ct_z = EvalAdd(slot_sum(EvalMult(ct_x, pt_w)), pt_b)        # z
ct_p = EvalPoly(ct_z, coeffs)                                # p(z), only if degree given
return [ct_z] or [ct_z, ct_p]
```

Depth (measured minimum, scale 50 bits, first modulus 60 bits): linear 1,
degree 1 → 2, degree 3 → 4, degree 5 → 4. Relinearization key generated only
for degree ≥ 2 (degree 1 works without it — verified).

### BGV circuit (`bgv.evaluate`)

Quantization (in `Quantizer.build`):

```
Sx = 2^8 = 256           x_q = round(x · Sx)                (CLIENT)
Sw = 2^10 = 1024         w_q = round(w · Sw)                (SERVER)
S  = Sx·Sw = 262144      b_q = round(b · S)
Sc = 2^9 = 512           a_k = round(c_k · Sc) for degree-1 coefficients  (largest Sc that fits)
t  = 1073872897          smallest prime ≥ 2^30 with t ≡ 1 (mod 2^16), 31 bits
```

Circuit:

```
ct_z = EvalAdd(slot_sum(EvalMult(ct_xq, pt_wq)), pt_bq)     # z_q,   scale S
ct_p = EvalAdd(EvalMult(ct_z, pt_a1), pt_(a0·S))            # P_q,   scale Sc·S   (degree 1 only)
client: z = z_q / S,  p = P_q / (Sc·S)
```

Wraparound safety: values are centered residues in [−(t−1)/2, (t−1)/2].
Because |x| ≤ 4 after clipping, `|x_q| ≤ 4·Sx` and the exact reachable maximum
is `zmax = 4·Sx·Σ|w_q| + |b_q|` = 3060738 (0.57 % of t/2). Degree 1 bound
`|a1|·zmax + |a0|·S` is 55.3 % of t/2; the boundary records reach 53.9 %.
`Quantizer.build` raises if either bound exceeds t/2. Only ciphertext–plaintext
products are used, so no relinearization key is generated. BGV uses OpenFHE's
default FLEXIBLEAUTO scaling technique.

## 6. File formats

### `artifacts/hf_plain.npz`

`X_train (239,11)`, `X_test (60,11)` (preprocessed, clipped), `y_train`,
`y_test`, `idx_train`, `idx_test` (row indices into the CSV), `w (11,)`,
`b ()`, `z_train`, `z_test`.

### `artifacts/hf_meta.json`

`dataset` (file, sha256, rows, target, excluded_columns) · `split` (seed,
test_size, stratified, n_train, n_test) · `preprocessing` (feature_order,
log1p_features, standardize_mean, standardize_scale, fitted_on, clip, order) ·
`model` (type, class_weight, C, C_grid, cv, cv_auc, weights, bias, score,
decision) · `metrics_test` (logistic_regression, random_forest_reference) ·
`scores` (z_train_range, z_test_range, worst_case_abs_z_under_clipping) ·
`polynomial` (form, fit_range, fit_margin, grid_points,
test_scores_outside_fit_range, degrees → {coefficients_ascending,
max_abs_err_on_fit_range, decision_agreement_with_sigmoid, test_metrics}) ·
`note` · `versions`.

### `demo/<run>/` (client/server exchange)

```
public/protocol.json          Protocol: scheme, degree, depth, batch, sum_method, feature_order,
                              log1p_features, standardize_mean, standardize_scale, clip,
                              BGV only: t, sx, s, sc
client_private/   (0700)      secret_key.bin (0600), public_key.bin, crypto_context.bin, decrypted.json
to_server/                    crypto_context.bin, rotation_keys.bin, [relin_key.bin], requests/0000.ct …
from_server/                  0000_z.ct, [0000_p.ct], …
```

All crypto objects are OpenFHE BINARY serializations. Request names are
sequence numbers only. `decrypted.json` holds per-request `z`, `decision`,
`p` (if degree) and, for BGV, `raw_residues`.

### `results/*.json`

- `environment.json` — machine and versions.
- `bgv_quant_sweep.json` — `{sc_bits, rows: [{sx_bits, sw_bits, z_max_err, agree,
  linear_bound, linear_bits, d1_err, d1_bits, d3_err, d3_bits, d5_err, d5_bits}]}`.
- `bgv_modulus_probe.json` — `{depth, security, rows: [{setting, t_bits, t,
  ring_dim, correct_runs, repeats} | {…, error}]}`.
- `benchmark_threads<N>_<sum>.json` — `{environment, omp_num_threads, slot_sum,
  repetitions, warmup_records_per_rep, timer, summary_statistic, configs: [...]}`;
  each config has `params`, `sizes_bytes` (crypto_context, public_key,
  secret_key_client_only, relin_key, rotation_keys, input_ciphertext,
  output_ciphertexts[]), `timing` (context, keygen, encrypt, evaluate, decrypt,
  latency_per_record → median/mean/std/p95/min/n; throughput), `memory`
  (peak_rss_mb_first_rep, peak_rss_mb_all_reps, rss_before_crypto_mb),
  `correctness`, `test_metrics_from_decrypted`.
- `client_server_demo.json` — per run: `steps` (timings per process), `sizes`,
  `audit`, `results`.
- `benchmark_tables.md` — generated; do not edit by hand.

## 7. Tests (`tests/`)

| File | Covers |
|---|---|
| `conftest.py` | `model` fixture; skips everything if `artifacts/` is missing |
| `test_model.py` | artifact consistency, split disjointness, odd polynomial structure |
| `test_bgv_quant.py` | batching prime, primality, bounds cover and are attained by boundary records, quantization preserves all test decisions |
| `test_roundtrip.py` | encrypt→evaluate→decrypt for CKKS (deg None/1/3 × both sum methods) and BGV (deg None/1 × both), on 3 test + 2 boundary records; BGV rejects degree 3 |
| `test_serialization.py` | full serialize → `ClearStaticMapsAndVectors()` → deserialize → evaluate → decrypt for CKKS 1/3 and BGV 1; client preprocessing reproduces `X_test` |
