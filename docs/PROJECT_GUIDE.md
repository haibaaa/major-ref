# Project Guide: Encrypted Logistic Regression

This guide teaches the project from the end-to-end workflow down to the
cryptographic algorithms. It is an orientation and learning document; the
architecture, runbook, experiment log, and threat model remain the detailed
technical records.

## 1. The project in one sentence

The project trains a logistic-regression model in plaintext, encrypts a
person's feature vector on a client, lets a server calculate a model score
without decrypting that vector, and compares four ways to do the encrypted
calculation: CKKS, BGV, BFV, and Paillier.

The main learning question is: **what do precision, supported arithmetic,
runtime, and ciphertext/key size cost for this small inference task?** This is
not encrypted training, and the repository is not a clinical decision system.

## 2. Project chunks and how data moves

| Chunk | What happens | Main files |
|---|---|---|
| A. Prepare the model | Load the heart-failure dataset, preprocess 11 features, split train/test, fit logistic regression, save weights and metadata. | `scripts/train_plain.py`, `src/fhe_compare/model.py`, `artifacts/` |
| B. Choose an encrypted circuit | Express the score and optional sigmoid approximation using operations supported by the selected encryption scheme. | `src/fhe_compare/ckks.py`, `bgv.py`, `bfv.py`, `paillier.py` |
| C. Client setup and encryption | Client publishes evaluation material and encrypts a preprocessed feature vector. It keeps the secret key. | `src/fhe_compare/client.py`, `protocol.py` |
| D. Server evaluation | Server loads ciphertext and plaintext model weights, calculates encrypted score (and optionally polynomial output), then returns ciphertext. | `src/fhe_compare/server.py` |
| E. Client decryption | Client decrypts the returned score and decides positive when `z > 0`. | `src/fhe_compare/client.py` |
| F. Compare and stress | Measure correctness, timings, memory and serialized sizes; test quantization near the decision boundary. | `scripts/benchmark.py`, `benchmark_paillier.py`, `margin_stress.py`, `results/` |

The high-level path is:

```text
plaintext dataset → preprocessing + model fit → public weights/model metadata
                                              ↓
client feature vector → encrypt → server computes encrypted score → ciphertext
          client secret key ← decrypt ← returned ciphertext
```

The server learns the model weights and metadata in this design. The secret
key stays with the client. Encryption hides the feature values from the server
under the scheme's security assumptions; it does not hide every possible
inference about a user, and this demo's byte audit is not a proof of security.

## 3. Logistic regression: the calculation being protected

For feature vector `x`, model weights `w`, and bias `b`, logistic regression
first computes a **logit**:

```text
z = w · x + b = Σ(wᵢ xᵢ) + b
```

The probability estimate is the sigmoid `σ(z) = 1 / (1 + exp(−z))`. Since the
sigmoid is increasing and `σ(0) = 0.5`, a 0.5 threshold is equivalent to the
simpler rule `z > 0`. This project can therefore make the binary decision from
the linear score alone. It optionally evaluates a polynomial approximation
`p(z)` to the sigmoid to show what nonlinear encrypted arithmetic costs.

Training happens before encryption using scikit-learn. At inference, the
server performs multiply-by-known-weight, sum, and bias addition. Each record
has 11 features, occupies slots 0–10 of a 16-slot ciphertext, and slot 0 of the
result is the score. Only one record is packed per ciphertext in this project.

The client repeats two `log1p` transforms, standardization using training
statistics, and clipping to `[-4, 4]`. Clipping is needed both to bound inputs
for safe integer arithmetic and to match the model's preprocessing. The
`time` column is excluded because it is follow-up duration, not an input
available at prediction time.

## 4. Homomorphic encryption in plain language

Ordinary encryption gives you a protected message but normally requires
decryption before calculation. **Homomorphic encryption (HE)** gives specific
ways to combine ciphertexts so that decrypting the result reveals the same
calculation on the original values. In this task, the server can calculate
`w · x + b` while `x` remains encrypted.

HE is not magic general-purpose encrypted computing: schemes allow particular
operations, and each operation consumes computational resources. Exact
integer schemes also calculate modulo a finite number, so values can wrap
around unless parameters and bounds are chosen correctly. The server still
sees ciphertext sizes, timing, public metadata, and the plaintext model.

### CKKS — approximate real-number arithmetic

**Idea:** CKKS encodes real numbers into polynomial-ring slots with a scale.
Homomorphic operations approximately preserve real arithmetic. It is designed
for useful numerical answers, not exact equality.

**In this project:** feature values and weights are encoded as approximate
reals. The server multiplies encrypted features by plaintext weights, sums the
slots, adds the bias, and may evaluate `p(z)` with `EvalPoly`. No manual integer
quantization is needed. Approximation error in the measured linear score was
around `10⁻¹³` to `10⁻¹¹` for these runs.

**When it is useful:** machine-learning inference, statistics, and signal
processing where small numeric error is acceptable; especially when values are
real-valued or the computation needs polynomial operations.

**Tradeoffs here:** supports the tested degree-1, degree-3, and degree-5
sigmoid approximations, but deeper polynomials cost more latency, larger
evaluation keys, and more multiplicative depth. It is an approximate scheme;
agreement on this finite test set does not mean all inputs get identical
decisions.

### BGV — exact modular integer arithmetic

**Idea:** BGV computes exactly on integers modulo a plaintext modulus `t`.
“Exact” means exact modular arithmetic on encoded integers; it does not mean
the original real-valued model is represented exactly. Real features and
weights must first be rounded to integers, which introduces quantization error.

The project uses scales `Sx` and `Sw`:

```text
xq = round(Sx · x)       wq = round(Sw · w)
score integer ≈ Σ(xqᵢ · wqᵢ) + round(Sx · Sw · b)
real score ≈ score integer / (Sx · Sw)
```

The server evaluates integer operations modulo `t`; the client converts
centered residues back to signed integers and divides by the scale. The
parameter builder checks that the largest score allowed by the clipped input
box stays below `t/2`, preventing modular wraparound from looking like a
plausible signed score.

**When it is useful:** workloads where exact integer operations matter, such
as integer statistics, counters, and fixed-point inference, and where scale
and value bounds can be planned. It can also be useful when exact modular
semantics are preferable to CKKS-style approximate arithmetic.

**Tradeoffs here:** quantization can move a score across zero, especially for
examples with a small margin `|z|`. The single-modulus parameter budget also
constrains how much precision and polynomial depth fit safely. The baseline
supports the linear score and degree-1 polynomial circuit; degree 3/5 were not
implemented in BGV for this comparison.

### BFV — another exact modular integer scheme

**Idea:** BFV is also an exact homomorphic scheme for modular integer
arithmetic. Like BGV, this project represents real numbers with scaled,
rounded integers, and like BGV its decoded real result can differ from the
original floating-point score because of that quantization.

**In this project:** BFV runs the same linear and degree-1 circuits with the
same quantized model as BGV. Comparing them helps isolate practical
implementation and parameter differences while holding the task and
quantization choices constant.

**When it is useful:** the same broad class of bounded integer or fixed-point
tasks as BGV. It is a reasonable alternative to evaluate when an application
needs exact integer operations and the library's BFV implementation fits its
security, performance, and parameter requirements.

**Tradeoffs here:** it does not remove quantization or modulus-bound concerns.
In the recorded settings OpenFHE selected ring dimension 8192 for BFV and
16384 for BGV; this project measured lower BFV latency for its tested paths.
That is a result for this configuration and machine, not a universal ranking
of BFV over BGV. BFV also has only linear and degree-1 coverage here.

### Paillier — additive public-key encryption

**Idea:** Paillier has an additive homomorphism. Given encryptions of `a` and
`b`, one can produce an encryption of `a + b`; raising an encryption to a
plaintext power produces an encryption of a plaintext multiple. Therefore,
with plaintext model weights the server can calculate an encrypted weighted
sum:

```text
Enc(x₁)^w₁ · Enc(x₂)^w₂ · … · Enc(xₙ)^wₙ · Enc(b) = Enc(Σwᵢxᵢ + b)
```

In the implementation features are encrypted separately. Paillier does not
provide the packed slots and general ciphertext-by-ciphertext multiplication
used for higher-degree polynomial evaluation in this project.

**When it is useful:** simple private sums, weighted sums, and aggregation
where one party holds the decryption key and the computation needs only
additions and multiplication by known plaintext constants.

**Tradeoffs here:** it is a useful simpler baseline for the linear score, but
not a drop-in replacement for BGV/BFV/CKKS when the circuit needs richer
homomorphic operations. A 512-bit run is only a functional smoke test and is
insecure; the measured 3072-bit run covered only five sampled records because
encryption was slow. The 3072-bit setting is a conventional approximate
security comparison point, not a formal equivalence to OpenFHE's parameter
standard.

## 5. What the project comparison says

| Scheme | Arithmetic used here | Linear score | Sigmoid polynomial here | Main precision issue |
|---|---|---:|---:|---|
| CKKS | Approximate real arithmetic | Yes | Degrees 1, 3, 5 | Approximation and encryption error |
| BGV | Exact modular integers + fixed-point encoding | Yes | Degree 1 | Rounding/quantization and modulus wraparound |
| BFV | Exact modular integers + fixed-point encoding | Yes | Degree 1 | Rounding/quantization and modulus wraparound |
| Paillier | Additions and plaintext scalar multiples | Yes | Affine degree 1 only | Fixed-point representation and integer bounds |

On the fixed 60-record test set, the implemented CKKS/BGV/BFV configurations
matched the plaintext decision on all records; the Paillier functional smoke
also agreed on all 60 at an insecure 512-bit key size. These are finite-set
results, not proof of universal correctness. The secure-size Paillier timing
experiment sampled five records and must not be presented as a full-set
benchmark.

## 6. The margin experiment and the project's contribution

The **margin** is the distance from a score to the decision boundary at zero:
`margin = |z|`. A score with a tiny margin is easy to flip if quantization or
approximation nudges it slightly. For fixed-point BGV/BFV, `margin_stress.py`
uses a conservative bound `B` on score rounding error. If `|z| > B`, the bound
guarantees that rounding alone cannot change the score's sign (assuming the
input clipping bound and arithmetic assumptions hold).

The script chose scales and a 33-bit plaintext modulus for BGV/BFV so the bound
(`0.001022`) fell below the minimum training margin (`0.001051`), then checked
train/test decisions and 54 synthetic labeled cases across three directions.
Both schemes had no flips in those checks at this setting. Lower-precision
settings did flip some synthetic decisions. The contribution to present is
the applied workflow—derive a bound from the model and clipping box, select
precision from the training margin, then test held-out and controlled cases.
It is not a new encryption algorithm, and the synthetic sweep should be
expanded before making broad reliability claims.

## 7. How to explore the repository

Start with [`status_and_next_steps.md`](status_and_next_steps.md) for the
project state, then use [`runbook.md`](runbook.md) for commands. Read
[`architecture.md`](architecture.md) before editing implementation and
[`threat_model.md`](threat_model.md) before making security claims. The
experiment methods and caveats are in [`experiments.md`](experiments.md), and
generated comparisons are in [`../results/benchmark_tables.md`](../results/benchmark_tables.md).

Typical commands from the repository root:

```bash
uv run python scripts/train_plain.py
uv run python scripts/benchmark.py --threads 1 --reps 3
uv run python scripts/benchmark_paillier.py --reps 3 --key-bits 3072 --records 5
uv run python scripts/margin_stress.py
uv run pytest
```

Training requires the dataset described in `docs/architecture.md`; it is not
committed. Use the runbook for required setup and the full regeneration order.
