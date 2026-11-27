# Experiment log

Each entry: purpose, command, what was measured, and the conclusion. All
numbers were produced in this repository on the development laptop (Intel
i7-7600U, 4 logical CPUs, Arch Linux, Python 3.12.12, OpenFHE 1.6.1.0.24.4)
unless stated. Single-run timings are indicative only; controlled timings are
in F-1.

## A — Environment

- Python candidates on the machine: 3.11, 3.13, 3.14 (system). uv downloaded
  3.12.12 after the wheel inspection showed a cpython-312-only extension.
- `CLAUDE.md`'s suggested `uv python pin 3.11` is **not usable** with
  OpenFHE 1.6.1 (requires ≥ 3.12, ships only a 3.12 extension).

## B — Plaintext model (`scripts/train_plain.py`)

The original `train_hf.py`, `hf_plain.npz`, `hf_meta.json`, `bgv_check.py` and
`ckks_smoke.py` were **not available**; training was rebuilt from the
decisions in `CLAUDE.md` §4.

Reproduced exactly against the user-reported historical run: best C = 0.1,
CV AUC 0.803, test accuracy 0.667, AUC 0.741, F1 0.500, recall 0.526,
confusion matrix `[[30, 11], [9, 10]]`, positive rate 0.321.

Differences from the historical run (settings that were not recorded):

| Item | Historical | Now | Why |
|---|---|---|---|
| Random forest | acc 0.750, AUC 0.794 | acc 0.700, AUC 0.787 | RF hyper-parameters unknown; ours: 300 trees, balanced, seed 42 |
| Fit range | ±3.64 | ±4.32 | rule unknown; ours `1.1 · max|z_train|` |
| Poly max err deg 1/3/5 | 0.1165 / 0.0303 / 0.0074 | 0.1422 / 0.0455 / 0.0138 | wider fit range |

Determinism: two consecutive runs produced identical arrays in `hf_plain.npz`.
Train/test indices disjoint; train means ≈ 0 (max |mean| 0.011 after clipping).

Polynomials outside the fit range (|z| up to the clip-box worst case 11.68):

| degree | leaves [0, 1] at |z| ≥ | decision flips at |z| ≥ |
|---|---|---|
| 1 | 3.43 | never (≤ 11.68) |
| 3 | 6.97 | **6.02** |
| 5 | 4.31 | never (≤ 11.68) |

Synthetic boundary records (every feature at ±4 aligned with sign(w)) give
z = 11.30 and −11.68. On them p(z) was: deg 1 → 2.146 / −1.201; deg 3 → −5.53
/ 7.316 (wrong side of 0.5); deg 5 → 42.86 / −50.31. Encrypted CKKS reproduced
these plaintext polynomial values, so this is a property of the polynomials,
not of encryption. (An initial flip-point script reported "never" for degree 3
because z = 0 gives exactly 0.5 and `argmax` returned index 0; fixed by
starting the grid at 1e-6.)

## C — Smoke tests

- CKKS (`ckks_smoke.py`): N = 16384, depth 2, scale 50: max abs error
  x+y 8.6e-14, x·y 4.9e-13, x·y·y 8.2e-13.
- BGV (`bgv_smoke.py`): N = 16384, t = 65537, depth 2: all results equal exact
  arithmetic mod t; the slot 300·300 wraps to 24463 as expected.

## D — CKKS

### D-1 Linear score (`scripts/ckks_linear.py`)

N = 16384, depth 1, scale 50, first mod 60, batch 16. On 60 records: score
error mean 1.3e-13, max 4.2e-13; agreement 60/60. EvalSum left the total in
all 16 slots (checked, max diff 4.6e-13). Single run: keygen 48.5 ms, encrypt
6.5 ms, evaluate 10.8 ms, decrypt 6.3 ms.

### D-2 Polynomial sigmoid (`scripts/ckks_poly.py`)

| degree | min depth | CKKS err vs plaintext poly (max) | poly err vs sigmoid (max, test) | agreement | median eval (single run) |
|---|---|---|---|---|---|
| 1 | 2 | 2.1e-13 | 0.099 | 60/60 | 20 ms |
| 3 | 4 | 2.5e-13 | 0.026 | 60/60 | 58 ms |
| 5 | 4 | 1.4e-11 | 0.007 | 60/60 | 79 ms |

## E — BGV

### E-1 Quantization sweep (`scripts/bgv_quant.py`, no encryption)

Sc fixed at 2^8. Bits = log2(2·bound + 1) needed in a centered modulus.
Linear bound is the exact reachable maximum over the clip box; polynomial
bounds are triangle-inequality upper bounds. Selected rows:

| Sx | Sw | max z err | agreement | linear bits | deg-1 bits | deg-3 bits | deg-5 bits |
|---|---|---|---|---|---|---|---|
| 2^4 | 2^4 | 1.4e-1 | 0.983 | 12.5 | 18.1 | 36.1 | 53.4 |
| 2^6 | 2^6 | 3.9e-2 | 1.000 | 16.5 | 22.1 | 48.2 | 73.4 |
| 2^6 | 2^8 | 1.5e-2 | **0.983** | 18.5 | 24.1 | 54.2 | 83.5 |
| 2^8 | 2^8 | 8.2e-3 | 1.000 | 20.5 | 26.1 | 60.2 | 93.5 |
| **2^8** | **2^10** | **3.7e-3** | 1.000 | **22.5** | 28.1 | 66.2 | 103.5 |
| 2^10 | 2^10 | 2.4e-3 | 1.000 | 24.5 | 30.1 | 72.2 | 113.5 |

Observations: agreement is not monotone in scale (2^6/2^8 loses one record);
only scales with max error below the smallest |z| (0.0056) guarantee test
agreement — 2^8/2^10 was chosen. With Sc = 2^8 the degree-3/5 errors stay near
0.045 because small coefficients (c3 ≈ −0.0058, c5 ≈ 0.00031) quantize to
almost nothing; better accuracy needs larger Sc, i.e. even more bits.

### E-2 Plaintext-modulus limit (`scripts/bgv_modulus_probe.py`)

See `openfhe_notes.md` §3 item 10. The 31-bit modulus is the baseline
configuration. A 33-bit batching prime succeeded in three recorded runs; the
tested 32-bit prime failed, as did tested values at 34 bits and above. This
does not establish a universal maximum. Degree 3 (≥ 36 bits, ~48+ at useful
accuracy) and degree 5 (≥ 53 bits) remain outside measured single-modulus
configurations; linear and degree 1 are supported.

### E-3 BGV inference (`scripts/bgv_linear.py`)

t = 1073872897 (31 bits), Sx = 2^8, Sw = 2^10, Sc = 2^9, depth 2, N = 16384.
All 60 test records and both boundary records decrypt to exactly the Python-int
result. z quantization error mean 1.2e-3, max 3.7e-3; degree-1 p error vs float
polynomial max 2.3e-3; agreement 60/60. Boundary records reach |z_q| = 0.6 %
and |P_q| = 53.9 % of t/2. Single run: encrypt 7.7 ms, evaluate 24.0 ms,
decrypt (2 ciphertexts) 2.8 ms.

## F — Controlled benchmark (`scripts/benchmark.py`)

Method: each configuration in its own subprocess; `OMP_NUM_THREADS` fixed;
5 repetitions, each with a fresh context and keys; 3 warm-up records per
repetition discarded; per-record `time.perf_counter()` timings for encrypt,
evaluate, decrypt; statistic = median over 5 × 60 samples (keygen/context:
median over 5); OpenFHE global maps cleared after each repetition; peak RSS =
`ru_maxrss` of the worker. Correctness checked on every repetition plus
boundary records.

### F-1 Primary: 1 thread, EvalSum (`results/benchmark_threads1_evalsum.json`)

| config | N | depth | keygen ms | encrypt ms | evaluate ms (p95) | decrypt ms | latency ms | rec/s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| ckks_linear | 16384 | 1 | 85.9 | 6.62 | 16.57 (19.06) | 10.23 | 33.49 | 29.1 | 189 |
| bgv_linear | 16384 | 1 | 82.0 | 6.62 | 16.76 (17.05) | 1.49 | 24.89 | 40.1 | 190 |
| ckks_deg1 | 16384 | 2 | 93.7 | 7.57 | 29.60 (31.23) | 21.80 | 58.90 | 17.0 | 198 |
| bgv_deg1 | 16384 | 2 | 96.5 | 8.11 | 35.30 (35.72) | 3.53 | 46.96 | 21.3 | 199 |
| ckks_deg3 | 16384 | 4 | 206.5 | 10.63 | 91.90 (92.80) | 28.32 | 130.89 | 7.6 | 231 |
| ckks_deg5 | 16384 | 4 | 212.1 | 10.59 | 114.31 (115.37) | 28.55 | 153.52 | 6.5 | 231 |

Sizes (identical for CKKS and BGV on the same circuit):

| circuit | public key | rotation keys | relin key | input ct | output ct(s) | secret key |
|---|---|---|---|---|---|---|
| linear | 0.75 MiB | 15.01 MiB | — | 769 KiB | 513 KiB | 385 KiB |
| degree 1 | 1.00 MiB | 18.01 MiB | — | 1025 KiB | 769 + 513 KiB | 513 KiB |
| degree 3/5 (CKKS) | 1.50 MiB | 36.01 MiB | 6.00 MiB | 1537 KiB | 1282 + 513 KiB | 769 KiB |

Correctness: CKKS max z error 5–8e-13, p vs plaintext polynomial ≤ 1e-11;
BGV exact on all test and boundary records, z quantization error 3.7e-3;
agreement 1.000 everywhere; test metrics from decrypted outputs identical to
plaintext in every configuration.

### F-2 Variants

- **4 threads** (`benchmark_threads4_evalsum.json`): evaluate ckks_linear 10.87,
  bgv_linear 11.51, ckks_deg1 21.93, bgv_deg1 25.19, ckks_deg3 62.60, ckks_deg5
  78.43 ms; decrypt somewhat faster (CKKS deg1 15.6 ms); encrypt unchanged.
- **Explicit rotations** (`benchmark_threads1_rotations.json`): rotation keys
  15.0 → 10.0 MiB (linear), 18.0 → 12.0 (deg 1), 36.0 → 24.0 (deg 3/5); evaluate
  +40–50 % (e.g. ckks_linear 23.16, bgv_deg1 47.10 ms); keygen faster.
- Microbenchmark (100 iterations, depth 1, 1 thread): summation alone EvalSum
  22.2 / 23.2 ms vs rotations 33.6 / 34.9 ms (CKKS / BGV).

### F-3 Methodology bugs found and fixed (do not reintroduce)

1. Peak memory accumulated across repetitions (first full run: up to 400 MB)
   because evaluation keys stay in OpenFHE global maps → clear after each
   repetition; report first-repetition peak.
2. Relinearization-key size reported as 9 bytes when none was generated (empty
   map) → report 0 for key sets not generated.
3. A docstring claimed EvalSum keys cover every ring rotation; measured 6 keys →
   corrected.

### Interpretation (measured, not generalised)

- Same circuit ⇒ same N, same limb count ⇒ identical key/ciphertext sizes.
- BGV's lower latency comes almost entirely from decryption; homomorphic
  evaluation is similar (BGV slightly slower for degree 1).
- Higher-degree sigmoid approximations cost CKKS 2.2–2.6× the degree-1 latency
  and double the rotation-key material, for lower approximation error (0.099 →
  0.026 → 0.007 max vs sigmoid on test).
- These are relative results on one machine; CPU frequency scaling was not
  pinned and the machine was not otherwise idle.

## G — Client/server demonstration (`scripts/demo_client_server.py`)

### G-1 Results (`results/client_server_demo.json`)

| run | agreement | setup bundle (once) | request / response per record | sk windows found | pk sent | preprocessed patterns found |
|---|---|---|---|---|---|---|
| ckks_deg1 | 60/60 | 18.0 MiB | 1025 / 1283 KiB | 0/6 | no | 0/180 |
| bgv_deg1 | 60/60, exact | 18.0 MiB | 1025 / 1283 KiB | 0/6 | no | 0/180 |
| ckks_deg3 | 60/60 | 42.0 MiB | 1538 / 1795 KiB | 0/6 | no | 0/180 |
| bgv_linear | 60/60, exact | 15.0 MiB | 769 / 513 KiB | 0/6 | no | 0/180 |

All requests within a run have identical byte size. Positive decisions 21 =
FP 11 + TP 10, matching plaintext.

### G-2 Byte-audit false positive (resolved)

The first audit searched requests for float64 encodings of raw values and
"found" up to 216 of 339 patterns. Investigation: all hits were raw round
numbers (e.g. 30.0, 1.0, 58.0 — six zero bytes + two bytes), some located in
serialization headers; no preprocessed (high-entropy) value matched. A control
— same patterns searched in a *different* record's request — gave the same
rate (CKKS 0.070 vs 0.061; BGV 0.120 vs 0.114), so the hits are
data-independent. The audit now reports preprocessed-value hits (0/180) and
raw round-value own-vs-control rates. The audit only rules out accidental
inclusion; it is not a confidentiality proof.

## H — Added schemes

### H-1 BFV

Command: `uv run python scripts/benchmark.py --threads 1 --reps 3 --only bfv_linear bfv_deg1 --output results/benchmark_threads1_bfv_evalsum.json`.
Both circuits used t = 1073872897, Sx = 2^8, Sw = 2^10, and the same quantized
model as BGV. OpenFHE selected N = 8192 at `HEStd_128_classic` (the CKKS/BGV
baseline uses N = 16384). On 60 test records and two synthetic edge records,
both outputs matched exact integer arithmetic and all test decisions matched
plaintext.

| circuit | median latency / record | encrypt | evaluate | decrypt | input ct | max z error |
|---|---:|---:|---:|---:|---:|---:|
| BFV linear | 9.59 ms | 3.94 ms | 4.88 ms | 0.72 ms | 257 KiB | 3.7e-3 |
| BFV degree 1 | 16.74 ms | 5.02 ms | 9.72 ms | 1.76 ms | 385 KiB | 3.7e-3 |

The separate-process client/server exchange also completed for both circuits;
each returned 21 positive decisions and agreed with the plaintext score on all
60 test records. These are three-repetition, single-machine measurements, not
a general performance claim.

### H-2 Paillier at two key sizes

Command: `uv run python scripts/benchmark_paillier.py --reps 1 --key-bits 512`.
The linear score decrypted with maximum error 4.4e-16 and decision agreement
60/60. This confirms the additive-encryption path works, but the 512-bit key is
insecure and is only a functional smoke check.

Command: `uv run python scripts/benchmark_paillier.py --reps 3 --key-bits 3072 --records 5`.
Five test records evenly spaced across the fixed test set were evaluated per
repetition. Maximum score error was 2.2e-16 and decision agreement was 5/5.
Median key generation was 2.94 s; encryption was 3.24 s per record, server
evaluation 94.7 ms, and decryption 83.5 ms. The request size estimate was about
20.4 KB per record. This uses a conventional approximate 128-bit modulus size,
but is not a formal security equivalence to OpenFHE's parameter standard.
Linear extrapolation puts encryption for all 60 records near 3.2 minutes;
that full run was not measured. Decimal ciphertext sizes are estimates, not a
binary wire-format measurement.

## I — Margin-aware precision and decision stability

Command: `uv run python scripts/margin_stress.py`.
The script constructs clipped feature vectors with 19 target logits from
−0.01 to +0.01, including exact zero, along the model weight direction. At
each target it tests the aligned vector and two 1.5-unit perturbations
orthogonal to the weight vector. Exact zero is a tie and is excluded from flip
counts, giving 54 labeled inputs per CKKS/BGV/BFV configuration. Paillier uses
the 18 aligned nonzero-margin vectors to limit runtime. It evaluates CKKS,
BGV, BFV and Paillier; BGV and BFV each use three baseline scales and a
matched higher-precision setting.
The JSON records target/observed logits, timing, flips, integer exactness, and
training/test agreement. The vectors cover three directions through the
decision boundary; they are synthetic stress cases, not patient samples.

### Margin-based scale rule

For nearest-integer quantization with input scale `Sx`, weight scale `Sw`, n
features and `|x_i| ≤ C`, each rounding error is at most 0.5 in integer space.
The real-score error is therefore bounded by:

```text
B = sum(|w_i|)/(2*Sx) + n*C/(2*Sw) + n/(4*Sx*Sw) + 1/(2*Sx*Sw)
```

The terms bound feature rounding, weight rounding, their product, and bias
rounding. The modulus must also hold the full clipped-domain score without
wraparound. If a plaintext score has margin `|z| > B`, the quantized score
keeps its sign. Scales are chosen from the minimum training-score margin, then
checked against the held-out test set and controlled synthetic margins.

The smallest training margin is 0.001051. For both BGV and BFV, `Sx=2^12`,
`Sw=2^15`, and the 33-bit batching prime `t=4295294977` give a bound of
0.001022, just below that margin; the clipped-domain linear bound uses 73.0%
of `t/2`. This parameter set was selected from the training margin, then checked
on held-out data: the bound covers observed train/test errors (maximum
0.000205), and all train and test decisions agree with plaintext. Neither
scheme flips any of the 54 synthetic labeled inputs at this setting. OpenFHE
selected N=16384 for BGV and N=8192 for BFV; median stress-path latency was
19.54 ms and 7.18 ms respectively.

| scheme / settings | score bound | training agreement | test agreement | flips / 54 synthetic inputs |
|---|---:|---:|---:|---:|
| CKKS, 50-bit scale | — | — | — | 0 / 54 |
| BGV, Sx=2^8, Sw=2^10, t=31 bits | 0.02711 | 1.000 | 1.000 | 2 |
| BFV, Sx=2^8, Sw=2^10, t=31 bits | 0.02711 | 1.000 | 1.000 | 2 |
| BGV, Sx=2^12, Sw=2^15, t=33 bits | **0.001022** | **1.000** | **1.000** | **0** |
| BFV, Sx=2^12, Sw=2^15, t=33 bits | **0.001022** | **1.000** | **1.000** | **0** |
| Paillier, 3072-bit modulus, aligned path | — | — | — | 0 / 18 |

Across tested BGV/BFV settings, observed flips were not monotonic in scale:
the 2^10/2^10 setting flipped 7 of 54 inputs while 2^8/2^10 flipped 2, even
though its largest observed error was lower. The selected 33-bit configuration
had no flips for either exact scheme. Report both an error bound and observed
decisions; higher scale alone did not guarantee reliability in these finite
stress cases.
Full per-setting values are in `results/margin_stress.json`. The 3072-bit
Paillier run is slow (about 3.3 s per record in this one-run experiment).
