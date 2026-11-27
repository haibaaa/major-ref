# Status and next steps

Phase letters follow `CLAUDE.md` §6.

## Phase status

| Phase | Status | Main outputs | Commit |
|---|---|---|---|
| A — repo and environment audit | **done** | `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `results/environment.json` | `d195c97` |
| B — plaintext artifacts | **done** (retrained; originals were not available) | `scripts/train_plain.py`, `artifacts/hf_plain.npz`, `artifacts/hf_meta.json` | `50927dd` |
| C — OpenFHE smoke tests | **done** | `scripts/ckks_smoke.py`, `scripts/bgv_smoke.py` | `d195c97` |
| D — CKKS inference | **done** (linear, degree 1/3/5) | `scripts/ckks_linear.py`, `scripts/ckks_poly.py`, `src/fhe_compare/ckks.py` | `8a2cd9e`, `6447b90` |
| E — BGV readiness and inference | **done** (linear, degree 1; degrees 3/5 infeasible at one modulus) | `scripts/bgv_quant.py`, `scripts/bgv_modulus_probe.py`, `scripts/bgv_linear.py`, `src/fhe_compare/bgv.py` | `f8d8d70`, `3da5aaf` |
| F — fair comparison | **done** | `scripts/benchmark.py`, `scripts/report.py`, `results/benchmark_*.json`, `results/benchmark_tables.md` | `27d19c7`, `e8d5f9c` |
| G — client/server demo | **done** | `src/fhe_compare/{client,server,protocol}.py`, `scripts/fhe_client.py`, `scripts/fhe_server.py`, `scripts/demo_client_server.py`, `docs/threat_model.md`, `results/client_server_demo.json` | `efb6871` |
| H — results and presentation | **not started** | — | — |

Tests: 22 pass (`uv run pytest`, ~8 s).

## Scheme extensions

BFV is integrated into the OpenFHE benchmark and client/server flow. Three-run
benchmarks and linear/degree-1 file exchanges now pass, including exact
quantized-reference and edge-record checks. BFV selected N=8192 at the
configured `HEStd_128_classic` level, while CKKS/BGV selected N=16384; compare
the measured outputs and parameters in `results/benchmark_threads1_bfv_evalsum.json`.

Paillier is implemented as a separate linear-score baseline using `phe`. The
512-bit smoke agrees on all 60 decisions but is insecure. A three-repetition
sample of five evenly spaced test records with 3072-bit keys completed with
5/5 decision agreement; full-set performance remains unmeasured. The baseline
is linear-only and does not use the OpenFHE exchange protocol.

## Focused project contribution

The current project contribution is a **margin-aware precision selection and
decision-stability evaluation**. A conservative rounding-error bound is
computed from the clipping box and quantization scales. The matched 33-bit
BGV/BFV setting has a bound of 0.001022, below the minimum training margin of
0.001051; it preserved all training and test decisions and had no flips among
54 synthetic labeled inputs across three directions. Lower-precision settings
flipped decisions, with counts that were not monotonic in scale. At the matched
parameters BFV used N=8192 versus BGV N=16384 and had lower measured stress-path
latency. Present this as an applied project contribution, not a new
cryptographic method. To strengthen it, sample more directions and margins,
then repeat the same rule on another split or dataset if time permits.

## Next step: Phase H (results and presentation)

1. **Plots**, generated only from saved JSON in `results/` (never hand-entered):
   - latency by scheme and operation (encrypt / evaluate / decrypt), 1 vs 4 threads;
   - serialized sizes (public key, rotation keys, relin key, input/output ciphertext);
   - error by configuration: CKKS encryption error vs BGV quantization error vs
     polynomial approximation error (log scale);
   - effect of polynomial degree (CKKS 1/3/5): latency, key size, max error vs sigmoid;
   - polynomial behaviour outside the fit range (p(z) for |z| ≤ 11.68 vs sigmoid,
     marking the fit range ±4.32 and degree-3 flip at 6.02);
   - BGV quantization trade-off from `results/bgv_quant_sweep.json` (bits vs error).
   Load the `dataviz` skill before writing any chart code. Use matplotlib (already
   a dependency), write PNG/SVG to `results/figures/`, one script e.g.
   `scripts/plots.py`.
2. **Top-level `README.md`** with exact setup and run instructions (it does not
   exist yet; CLAUDE.md §8 requires it). `docs/runbook.md` is the source to condense.
3. **Results and limitations write-up** (Markdown in `docs/` or `results/`),
   drawing numbers from the generated tables; reuse `docs/threat_model.md`.

## Open items and known gaps

- **BGV degree 3/5 via CRT** (optional extension, decision D-09): run the circuit
  under 2–3 coprime batching plaintext moduli (~30 bits each) and recombine on the
  client with the Chinese Remainder Theorem. Requires larger coefficient scale Sc
  than 2^8 (see `experiments.md` E-2) and multiplies BGV cost by the number of moduli.
- **Timing data-dependence** was not analysed (threat model lists it as not measured).
- **Benchmark controls**: CPU frequency scaling/turbo not pinned, machine not idle.
  Timings are relative comparisons on one laptop (i7-7600U), not absolute figures.
- **Why CKKS decryption is slower than BGV** (≈10–28 ms vs 1.5–3.5 ms) was not
  investigated. Do not explain it in reports without measuring.
- **Why OpenFHE's EvalSum is faster than explicit rotate-and-add** (≈22 vs 34 ms)
  was not investigated.
- **Exploratory scripts duplicate package logic** (`ckks_linear.py`, `ckks_poly.py`,
  `bgv_linear.py` predate `src/fhe_compare`). They are kept as the record of
  Phases D/E. The package is canonical; do not "fix" results by editing the old
  scripts. Refactoring them onto the package is optional.
- **Multi-record packing** (several patients per ciphertext) was not explored;
  every configuration uses one record per ciphertext, 16 slots.
- **Ring dimension 8192** is rejected by OpenFHE for these settings at 128-bit
  security; smaller parameters were not explored further (e.g. lowering
  `FirstModSize`/`ScalingModSize`), and would need accuracy re-validation.
- `results/environment.json` was regenerated before the final benchmark; if you
  rerun benchmarks on another machine, rerun `scripts/record_env.py` first and
  never mix results from different machines in one table.
