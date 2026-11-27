# Runbook

Exact commands for setting up, regenerating and verifying everything. All
commands run from the repository root. Runtimes are approximate wall-clock
observations on the development laptop (Intel i7-7600U, 4 logical CPUs), not
benchmarks.

## 1. Environment

| Item | Value | Why |
|---|---|---|
| OS | Linux (developed on Arch Linux, kernel 7.2, glibc 2.44) | target is native Linux or WSL2 |
| uv | 0.10.2 | project/dependency manager (`CLAUDE.md` §3) |
| Python | **3.12** (pinned in `.python-version`, `requires-python = ">=3.12,<3.13"`) | the OpenFHE wheel ships `openfhe.cpython-312-x86_64-linux-gnu.so` only, despite its `py3-none-any` tag; on 3.13/3.14 it installs but cannot import |
| OpenFHE | `openfhe==1.6.1.0.24.4` (exact pin) | the `.24.4` build (Ubuntu 24.04 toolchain) loads on glibc 2.44; `.22.4` builds also exist on PyPI |
| Other deps | numpy, pandas, scikit-learn, matplotlib, joblib, phe (Paillier); dev: pytest | see `pyproject.toml` / `uv.lock` |
| Build backend | `uv_build` | makes `src/fhe_compare` installable so scripts can `import fhe_compare` |

Setup on a fresh clone:

```bash
uv sync                      # creates .venv, installs locked deps + fhe_compare
uv run python -c "import openfhe; print(openfhe.__file__)"   # must import, not just install
uv run python scripts/ckks_smoke.py && uv run python scripts/bgv_smoke.py
```

Do not use pip, do not create another virtualenv, do not install globally.

### Shell quirk on the development machine

`ls` is aliased to `eza --icons=always`, which **hangs when there is no TTY**
(e.g. inside an agent's Bash tool). Use `command ls`, `find`, or `stat`.

## 2. Data

- The dataset is **not** in git (`data/*.csv` is ignored). Place it at
  `data/Heart_failure_clinical_records_dataset.csv` — note the capital **H**;
  `CLAUDE.md` mentions a lowercase name, the code uses the real one.
- Expected: 299 rows × 13 columns, no missing values, target `DEATH_EVENT`
  (positive rate 0.321). `train_plain.py` asserts the exact column order.
- SHA-256 prefix of the file used for all results: `085c4740178a5255`
  (full hash in `artifacts/hf_meta.json` → `dataset.sha256`).
- Never print rows. Schema checks should print shape, columns, dtypes, rates only.

## 3. Regeneration order (dependencies flow downward)

| Step | Command | Produces | Approx. time |
|---|---|---|---|
| 1 | `uv run python scripts/record_env.py` | `results/environment.json` | < 1 s |
| 2 | `uv run python scripts/train_plain.py` | `artifacts/hf_plain.npz`, `artifacts/hf_meta.json` (deterministic: identical arrays on rerun) | seconds |
| 3 | `uv run pytest` | 22 tests (skip if artifacts missing) | ~8 s |
| 4 | `uv run python scripts/ckks_smoke.py`, `scripts/bgv_smoke.py` | console only, exit 0 = PASS | seconds |
| 5 | `uv run python scripts/ckks_linear.py` | console; exit 0 = PASS | seconds |
| 6 | `uv run python scripts/ckks_poly.py` | console (depth search + boundary test) | ~1 min |
| 7 | `uv run python scripts/bgv_quant.py` | `results/bgv_quant_sweep.json` | seconds |
| 8 | `uv run python scripts/bgv_modulus_probe.py` | `results/bgv_modulus_probe.json` | ~1–2 min |
| 9 | `uv run python scripts/bgv_linear.py` | console; exit 0 = PASS | seconds |
| 10 | `uv run python scripts/benchmark.py --threads 1 --reps 5` | `results/benchmark_threads1_evalsum.json` | ~3.5 min |
| 11 | `uv run python scripts/benchmark.py --threads 1 --reps 5 --sum rotations` | `results/benchmark_threads1_rotations.json` | ~4 min |
| 12 | `uv run python scripts/benchmark.py --threads 4 --reps 5` | `results/benchmark_threads4_evalsum.json` | ~2.5 min |
| 13 | `uv run python scripts/report.py` | `results/benchmark_tables.md` (from every `benchmark_threads*.json`) | < 1 s |
| 14 | `uv run python scripts/demo_client_server.py --runs ckks:1 bgv:1 bfv:1 ckks:3 bgv:0` | `demo/<run>/…` (ignored), `results/client_server_demo.json` | ~1 min |
| 15 | `uv run python scripts/benchmark_paillier.py --reps 3 --key-bits 3072 --records 5` | `results/benchmark_paillier_3072b_5records.json` | ~1 minute |
| 16 | `uv run python scripts/margin_stress.py` | `results/margin_stress.json` | ~1–2 minutes |

If you retrain (step 2) you must rerun everything after it: BGV scales/bounds,
polynomial coefficients and every result depend on the weights.

Long runs (steps 10–12) exceed a 2-minute tool timeout; run them in the
background and wait for completion rather than polling.

## 4. Command reference

### Benchmark (`scripts/benchmark.py`)

```
--threads N        OMP_NUM_THREADS for the worker processes (default 1)
--reps R           repetitions; each = new context + keys + all 60 records (default 5)
--sum METHOD       evalsum (default) | rotations  — slot-summation method
--only NAME ...    subset of: ckks_linear bgv_linear bfv_linear ckks_deg1 bgv_deg1 bfv_deg1 ckks_deg3 ckks_deg5
--worker NAME      internal: run one config and print JSON (used by the parent)
```

Output file name: `results/benchmark_threads<N>_<sum>.json`. Running a subset
with `--only` **overwrites** that file with only the subset — rerun the full set
before reporting.

### Client/server (`scripts/fhe_server.py`, `scripts/fhe_client.py`)

Manual step-by-step run (the driver does exactly this, each as its own process):

```bash
R=demo/ckks_deg1
uv run python scripts/fhe_server.py init     --run $R --scheme ckks --degree 1   # omit --degree for linear
uv run python scripts/fhe_client.py setup    --run $R
uv run python scripts/fhe_client.py encrypt  --run $R
uv run python scripts/fhe_server.py evaluate --run $R
uv run python scripts/fhe_client.py decrypt  --run $R
```

Driver: `scripts/demo_client_server.py --runs scheme:degree ...` where degree
0 means linear. Valid: `ckks:0 ckks:1 ckks:3 ckks:5 bgv:0 bgv:1 bfv:0 bfv:1`. The driver
deletes and recreates `demo/<run>/`, chmods `client_private/` to 000 during the
server step, audits server-visible files and compares with plaintext.

## 5. Verification checklist after any change

- `uv run pytest` — all pass.
- Encrypted changes: run the relevant script or `benchmark.py --reps 1` and
  confirm agreement 1.000, BGV/BFV `decrypted_equals_exact_integer: true`, CKKS
  errors ~1e-12 or smaller.
- Client/server changes: run the demo; confirm 0 secret-key windows found, public
  key not sent, 0 preprocessed feature patterns found.
- Report the exact command and outcome (CLAUDE.md §2).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError` for openfhe | wrong Python; must be 3.12 (`uv run python --version`) |
| `EvalSumKeyGen operation has not been enabled. Enable(ADVANCEDSHE)…` | enable `PKESchemeFeature.ADVANCEDSHE` on the context |
| `The specified ring dimension (8192) does not comply with HE standards recommendation (16384)` | OpenFHE enforces the HE-standard minimum; let it choose N |
| `Decode(): The decryption failed because the approximation error is too high` (CKKS) | multiplicative depth too small for the circuit |
| `DropLastElement(): Removing last element of DCRTPoly renders it invalid` / `SetLevel(): The current multiplicative depth [k] is insufficient` | depth too small |
| BGV `We do not support moduli greater than 60 bits` or `EstimateLogP() failure` | plaintext modulus too large (see `openfhe_notes.md`) |
| BGV result wrong but no error | possible FIXEDMANUAL with t > 32 bits, or wraparound — compare with exact Python-int arithmetic |
| Peak memory grows across repetitions | OpenFHE global key maps accumulate; call `CryptoContext.ClearStaticMapsAndVectors()` |
| A shell command hangs with no output | the `ls`→`eza` alias; use `command ls` |
| Tests skipped | `artifacts/` missing; run `scripts/train_plain.py` |
