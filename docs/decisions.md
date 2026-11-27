# Decision log

Format: decision · rationale · alternatives considered · who decided. "User"
means the user explicitly approved; "agent" means the agent chose it under a
general instruction (e.g. "proceed as you see fit") and reported it. Re-open a
decision only with new evidence, and ask the user before changing anything
listed as user-approved or anything affecting scope (CLAUDE.md §2).

## Environment and process

**D-01 Python 3.12, not 3.11.** The OpenFHE 1.6.1 wheel requires ≥ 3.12 and
contains only a CPython-3.12 extension. `requires-python = ">=3.12,<3.13"` so
nobody resolves on 3.13+. *Alternative:* older OpenFHE releases for 3.11 — not
pursued (no reason to use an older library). Agent; reported to user.

**D-02 Exact pin `openfhe==1.6.1.0.24.4`.** Reproducibility; the `.24.4` build
loads on this glibc. Agent.

**D-03 Git and commits.** User asked for `git init` and approved committing.
Commits per milestone, message ending with the `Co-Authored-By` line required
by the session. Never commit `data/`, `artifacts/`, `demo/`, keys or ciphertexts.

**D-04 Work in phases, stop at milestones.** CLAUDE.md §9. The user has
repeatedly said "proceed as you see fit / according to your recommendation";
that covers choices inside a phase, not scope changes.

## Data and model

**D-05 Retrain rather than reuse.** The user confirmed no original files exist.
`train_plain.py` rebuilds training from CLAUDE.md §4 and reproduces the
historical logistic-regression numbers exactly. User.

**D-06 Unrecorded training details chosen by the agent.**
`StratifiedKFold(5, shuffle=True, random_state=42)`; random forest 300 trees,
balanced, seed 42 (reference only); polynomial fit range `1.1 · max|z_train|`
(= ±4.32; the historical ±3.64 rule is unknown); least-squares fit on a
20001-point grid. Agent; reported.

**D-07 Dataset filename.** The real file is
`Heart_failure_clinical_records_dataset.csv` (capital H). Code uses the real
name; the file was not renamed. Agent.

## Circuit design

**D-08 One record per ciphertext, 16 slots, slot 0 result.** Simplest correct
layout for a single-patient inference; batching several patients per
ciphertext is unexplored. Agent.

**D-09 BGV: linear + degree 1 only in the main comparison; degrees 3/5
reported as a limitation; CRT optional.** The baseline uses a 31-bit plaintext
modulus; a 33-bit batching prime passed the recorded three-run probe, while
the tested 32-bit prime and values at 34 bits or above failed. Degree 3 needs
≥ 36 (≈ 48+ at useful accuracy), degree 5 ≥ 53. CRT across 2–3 moduli would
work but multiplies cost and complexity for the viva.
*Alternatives:* (A) implement CRT now; (C) tighter clipping (rejected — changes
preprocessing/model). User approved the agent's recommendation (B, CRT
optional).

**D-10 Decision from sign(z); p(z) is only a score.** Polynomials leave [0, 1]
outside the fit range and degree 3 inverts decisions for |z| ≥ 6.02, which
clipped inputs can reach (worst case 11.68). The server returns both z and
p(z); the client decides from z. *Alternatives:* fit over ±11.68 (much worse
in-range error); keep and document only. User ("keep it as you like" after
the agent recommended this option).

**D-11 Weights are plaintext on the server.** The server is the model owner;
only patient data is encrypted. This means ct×pt products only (cheaper, no
relinearization for linear/degree 1) and is stated in the threat model,
including the model-extraction risk.

**D-12 CKKS parameters.** Scale 50 bits, first modulus 60 bits, OpenFHE-chosen
N (16384), `HEStd_128_classic`; minimum depth per degree found by trial
(1/2/4/4). Not explored: smaller scale/first-mod to reach N = 8192. Agent.

**D-13 BGV parameters.** Sx = 2^8, Sw = 2^10 (smallest tested pair whose max z
error 3.7e-3 is below the smallest test |z| 0.0056); t = smallest 31-bit prime
≡ 1 mod 2^16 (1073872897); Sc = largest power of two whose degree-1 bound fits
(2^9); default FLEXIBLEAUTO scaling, because FIXEDMANUAL silently decrypted
wrong values above 32 bits. The linear BGV configuration reuses the same t
although it needs only ~23 bits — a smaller t was not tried. Agent.

**D-14 Minimal key material.** Relinearization key only for CKKS degree ≥ 2;
the public key is not sent to the server (not needed for ct×pt circuits).
Agent.

**D-15 Slot summation: EvalSum by default; explicit rotations as a measured
variant.** EvalSum is ~11 ms faster per record; rotations cut rotation-key
material by a third (6 → 4 keys). Per-record latency matters on every request,
keys are sent once. Both are in the benchmark (`--sum`). Agent.

## Measurement

**D-16 Benchmark methodology.** Subprocess per configuration (isolates OpenFHE
global state and peak RSS); `OMP_NUM_THREADS=1` as primary, 4 threads as a
variant; 5 repetitions with fresh keys; 3 warm-up records per repetition;
median (plus mean/std/p95/min) over repetitions × records; global key maps
cleared per repetition. Agent.

**D-17 Results come from saved JSON.** Tables (`report.py`) and future plots are
generated from `results/*.json`, never typed by hand (CLAUDE.md Phase H).

## Client/server

**D-18 File-based exchange, separate OS processes, no network.** CLAUDE.md
says a CLI demo is sufficient. The server reads only `public/` and
`to_server/`; `client_private/` is `chmod 000` while the server runs.

**D-19 Server publishes the protocol.** As model owner it computes BGV scales
and t from the model and publishes them with the preprocessing statistics in
`protocol.json`; the client receives scales needed to quantize/dequantize but
not the weights.

**D-20 Byte audit design.** Check secret-key byte windows, public-key absence,
and high-entropy preprocessed feature values; report raw round values with an
other-record control (see `experiments.md` G-2). State clearly that the audit
is not a confidentiality proof.
