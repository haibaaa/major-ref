# Project documentation — start here

This folder is the working knowledge base for anyone (human or AI agent)
continuing the **CKKS vs BGV encrypted logistic-regression** project. It
records what exists, how it works, what was measured, what was decided and
why, and which traps have already been found. `CLAUDE.md` in the repository
root remains the authoritative statement of goals, rules and phase order; these
documents describe the *current state* produced by following it.

Last updated: 2026-11-22, at commit `efb6871` (Phase G complete).

## Reading order

| # | Document | Read it when |
|---|---|---|
| 1 | [`status_and_next_steps.md`](status_and_next_steps.md) | Always first: phase status, open work, what to do next |
| 2 | [`runbook.md`](runbook.md) | Before running anything: setup, every command, runtimes, regeneration order |
| 3 | [`architecture.md`](architecture.md) | Before changing code: layout, modules, circuits, slot layouts, file formats |
| 4 | [`openfhe_notes.md`](openfhe_notes.md) | Before touching any OpenFHE call: verified API behaviour and pitfalls |
| 5 | [`experiments.md`](experiments.md) | When reporting or comparing results: every experiment, command and measured outcome |
| 6 | [`decisions.md`](decisions.md) | Before re-opening a design question: decision log with rationale and approvals |
| 7 | [`threat_model.md`](threat_model.md) | Before making any security or privacy claim |

## The project in one paragraph

A logistic-regression model (11 features, `time` excluded) is trained in
plaintext on the public Heart Failure Clinical Records dataset (299 rows,
80/20 stratified split, seed 42). The same model is then evaluated on
encrypted inputs with **CKKS** (approximate real arithmetic) and **BGV**
(exact modular integer arithmetic, requiring quantization), both through
OpenFHE 1.6.1's Python bindings. The two are compared for correctness,
prediction agreement, latency per operation, memory, and key/ciphertext sizes,
and a file-based client/server demonstration shows the server evaluating
without the secret key or plaintext features. The goal is to measure
trade-offs, **not** to show encrypted inference is more accurate, and not to
build a clinical system.

## Headline facts (all measured in this repository)

- Plaintext test metrics (60 records): accuracy 0.667, ROC AUC 0.741, F1 0.500,
  recall 0.526, confusion matrix `[[30, 11], [9, 10]]`; best C = 0.1, CV AUC 0.803.
- Every encrypted configuration agrees with the plaintext decision on 60/60 records.
- CKKS score error ≈ 1e-13 to 1e-11; BGV dequantization error on z ≤ 3.7e-3 and
  BGV results equal the exact integer computation (test and boundary records).
- Ring dimension 16384 for every configuration at `HEStd_128_classic`.
- BGV supports only the linear and degree-1 circuits with a single plaintext
  modulus (~31 bits usable); degrees 3 and 5 are CKKS-only. This is a reported
  limitation, by decision (see `decisions.md`, D-09).
- Primary benchmark (1 thread, EvalSum): latency per record CKKS linear 33.5 ms,
  BGV linear 24.9 ms, CKKS deg-1 58.9 ms, BGV deg-1 47.0 ms, CKKS deg-3 130.9 ms,
  CKKS deg-5 153.5 ms. Single laptop CPU; see `experiments.md` for caveats.

## Rules that matter most for a new agent

1. Use `uv run …` for everything; Python is pinned to **3.12** because the
   OpenFHE wheel's native extension is CPython-3.12-only.
2. Never print, log or commit raw patient rows. `data/*.csv`, `artifacts/` and
   `demo/` are git-ignored; `demo/` contains secret keys.
3. Do not change the split, preprocessing, feature order, weights or
   polynomial coefficients between CKKS and BGV runs. Retraining is
   deterministic, but every downstream result depends on it.
4. Verify OpenFHE behaviour empirically: a context that builds is **not**
   evidence that parameters are correct (see `openfhe_notes.md`, BGV
   FIXEDMANUAL silent wrong results).
5. Separate measured results from estimates in everything you write.
6. Work one phase at a time and stop for user review at milestones (CLAUDE.md §9).
7. In this machine's shell, `ls` is aliased to `eza` and **hangs without a TTY**;
   use `command ls`.
