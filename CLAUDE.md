# CLAUDE.md --- Encrypted Logistic-Regression Scheme Comparison

## 1. Project objective

Build an experimental comparison of **CKKS, BGV and BFV fully homomorphic
encryption (FHE)** for privacy-preserving machine-learning inference, with
Paillier as an additive-encryption baseline for the linear score.

The project uses a heart-failure clinical-records dataset and logistic
regression as the primary model. The intended workflow is:

1.  Train and evaluate a plaintext reference model.
2.  Implement encrypted inference with CKKS.
3.  Implement the corresponding encrypted inference with BGV and BFV.
4.  Compare Paillier on the linear score, and compare CKKS/BGV/BFV on
    correctness, predictive performance, runtime, memory,
    ciphertext/key sizes, and parameter requirements.
5.  Demonstrate a client/server arrangement in which the client retains
    the secret key and the server evaluates encrypted inputs.

The goal is **not** to show that encrypted ML is more accurate than
plaintext ML. The goal is to measure the privacy/computation trade-offs
and determine how CKKS and BGV behave on the same task.

## 2. How Claude should work

Act as a careful research and engineering assistant. Work incrementally
and explain important decisions.

-   Inspect the existing repository and files before changing anything.
-   Do not overwrite working scripts, trained artifacts, or user data
    without asking.
-   Prefer small, testable changes over a large one-shot implementation.
-   After each meaningful change, run the relevant tests or a small
    smoke test and report the exact command and outcome.
-   If a command fails, diagnose the actual error. Do not silently work
    around it or claim success.
-   Do not invent benchmark results, paper claims, API methods,
    parameter guarantees, or dataset properties.
-   Distinguish measured results from estimates and hypotheses.
-   When an OpenFHE Python API detail is uncertain, inspect the
    installed package, its examples, or version-matched documentation
    before writing code that depends on it.
-   Keep the implementation understandable enough for a university
    project presentation and viva.
-   Avoid unnecessary abstractions, frameworks, services, or
    dependencies.
-   Do not require a GPU. This project is designed to run on a CPU.
-   Ask before making a major change to the research scope, dataset,
    model, or cryptographic scheme.

## 3. Environment and tooling

Use **uv** for Python project and dependency management.

-   Declare project metadata and dependencies in `pyproject.toml`.
-   Commit and maintain `uv.lock`.
-   Use `uv run ...` for project commands.
-   Do not create or use a separate `pip`-managed virtual environment.
-   Do not install packages globally.
-   Prefer stable, mutually compatible dependency versions.
-   The target environment is Linux (native Linux or WSL2 Ubuntu).
-   Use Python version supported by the selected OpenFHE Python
    distribution. Verify wheel availability and importability before
    fixing the project's Python version.
-   OpenFHE's Python bindings are a critical dependency. Do not assume
    that a successful `uv add openfhe` means the native extension is
    usable. Verify with an actual import and a small cryptographic
    operation.

Typical initial setup (adapt after inspecting the machine and package
availability):

``` bash
uv init
uv python pin 3.11
uv add numpy pandas scikit-learn matplotlib joblib
uv add openfhe
uv run python -c "import openfhe; print(openfhe.__file__)"
```

If the OpenFHE package cannot be installed or imported with the selected
Python version/platform, stop and report the exact error. Investigate
supported Python versions and Linux wheels before proposing a source
build or alternative library. Do not silently replace OpenFHE with
another library: using one library for both CKKS and BGV is preferred
for a fair comparison.

Useful project commands:

``` bash
uv sync
uv run python --version
uv run python path/to/script.py
uv run pytest
```

Only add `pytest` or other development dependencies if tests are
actually being created.

## 4. Current project status and known decisions

The following work has already been completed in the prior workflow.
Inspect the repository to confirm which files are present; do not assume
every artifact has already been copied into it.

### Dataset

-   Dataset: **Heart Failure Clinical Records**.
-   Expected original CSV filename:
    `heart_failure_clinical_records_dataset.csv`.
-   Dataset size: 299 records, 13 original columns.
-   Target: `DEATH_EVENT`.
-   The `time` column is excluded from the primary model because it
    represents follow-up duration and may encode information only
    available after the prediction point.
-   The primary feature set therefore contains 11 features.

### Existing plaintext training result

A plaintext training script was run successfully. Its reported output
was:

``` text
shape: (299, 13) | missing: 0 | positive rate: 0.321
features: ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
best C: 0.1 | CV AUC: 0.803
logistic regression    acc=0.667 auc=0.741 f1=0.500 recall=0.526
confusion matrix [[TN FP],[FN TP]]:
 [[30 11]
  [ 9 10]]
random forest (ref)    acc=0.750 auc=0.794 f1=0.545 recall=0.474
z range train [-3.15, 3.93] | fit range +/-3.64
poly degree 1          acc=0.667 auc=0.741 f1=0.500 recall=0.526
   max err on fit range=0.1165 | agreement with true sigmoid=1.000
poly degree 3          acc=0.667 auc=0.741 f1=0.500 recall=0.526
   max err on fit range=0.0303 | agreement with true sigmoid=1.000
poly degree 5          acc=0.667 auc=0.741 f1=0.500 recall=0.526
   max err on fit range=0.0074 | agreement with true sigmoid=1.000
```

These numbers are user-reported results from a previous run, not results
independently reproduced by Claude. Preserve them as historical
reference until the script is rerun in the current environment.

### Existing artifact names

The earlier workflow produced or planned these files:

-   `hf_plain.npz` --- processed train/test arrays, labels, and trained
    logistic-regression weights.
-   `hf_meta.json` --- preprocessing metadata and polynomial
    approximation information.
-   `train_hf.py` --- plaintext training script.
-   `bgv_check.py` --- integer-quantization readiness experiment.
-   `ckks_smoke.py` --- initial CKKS API smoke test.

Search for these files before creating replacements. If `hf_plain.npz`
and `hf_meta.json` exist and are valid, reuse them rather than
retraining unnecessarily.

### Model and preprocessing decisions

-   Primary model: logistic regression.
-   `class_weight="balanced"` was used.
-   `C` was selected using stratified 5-fold cross-validation over
    `[0.01, 0.03, 0.1, 0.3, 1, 3]`, optimizing ROC AUC.
-   Train/test split: 80/20, stratified, seed 42.
-   The two heavily skewed features `creatinine_phosphokinase` and
    `serum_creatinine` receive `log1p`.
-   Standardization is fitted on training data only.
-   Transformed features are clipped to `[-4, 4]`.
-   The logistic-regression decision score is `z = X @ w + b`.
-   Candidate sigmoid approximations have degrees 1, 3, and 5. The
    previous script fitted odd polynomials to the sigmoid over a bounded
    interval.
-   Degree 1 is useful for classification decisions but is not a
    well-calibrated probability approximation.
-   Degree 3 or 5 may provide a smoother score at additional
    multiplicative depth and cost.
-   Do not call the outputs calibrated clinical probabilities. This is a
    technical demonstration, not a clinical decision-support system.

Important: independently verify that the polynomial coefficients, fit
interval, and saved metadata are consistent with the current model. Do
not assume a coefficient set from an earlier run is valid after
retraining.

## 5. Required architecture

Keep the design conceptually separated into two roles.

### Client

The client:

-   owns and generates the secret key;
-   holds the secret key and never sends it to the evaluation server;
-   preprocesses a record using the agreed preprocessing parameters;
-   encrypts the feature vector;
-   sends the ciphertext and only the public/evaluation material
    required for evaluation;
-   decrypts the returned encrypted result;
-   interprets the decrypted score or decision.

### Evaluation server

The server:

-   receives encrypted feature data;
-   receives public and evaluation keys only when required by the
    scheme;
-   evaluates the agreed logistic-regression circuit on ciphertexts;
-   returns an encrypted result;
-   never receives the secret key or plaintext patient feature vector.

For the initial implementation, these roles may be represented by
separate modules or scripts and serialized files. A networked web
application is optional and must not delay the cryptographic comparison.

Document what the implementation does and does not protect. Ciphertext
sizes, message sizes, access patterns, and timing may leak metadata.
Homomorphic encryption alone does not guarantee that the server returns
a correct result, nor does it solve authentication, authorization,
secure key storage, or malicious-client problems.

## 6. Implementation plan

Follow this order. Do not jump ahead while an earlier stage is failing.

### Phase A --- Repository and environment audit

1.  Inspect the directory tree, existing scripts, data files, and
    generated artifacts.
2.  Identify the actual CSV filename and its location.
3.  Inspect the current Python version, Linux distribution, and
    available OpenFHE package.
4.  Create or update `pyproject.toml` using uv without destroying
    existing configuration.
5.  Run `uv sync`.
6.  Verify `import openfhe`.
7.  Record Python, uv, OpenFHE, OS, and CPU details for reproducibility.

Do not print or commit private patient data. The dataset is small, but
treat it as sensitive in the project architecture and avoid adding raw
rows to logs or screenshots.

### Phase B --- Validate plaintext artifacts

1.  Confirm the CSV columns and target.
2.  Confirm that `time` is excluded in the primary experiment.
3.  Confirm the fixed split and preprocessing are deterministic.
4.  Confirm that preprocessing is fitted on training data only.
5.  Confirm that the saved weights have the expected 11-feature
    dimension.
6.  Recompute plaintext test predictions from the saved weights.
7.  Validate polynomial approximations over the actual score range.
8.  Save a clear experiment metadata file containing the seed, feature
    order, preprocessing parameters, model parameters, and polynomial
    coefficients.

Do not change the train/test split between CKKS and BGV experiments.

### Phase C --- OpenFHE smoke tests

Before implementing the full model:

1.  Create a minimal CKKS context.
2.  Generate keys.
3.  Encrypt a short vector.
4.  Perform a simple supported operation, such as ciphertext
    multiplication.
5.  Decrypt and compare against the expected approximate result.
6.  Record the observed numerical error.
7.  Separately create a minimal BGV context.
8.  Test encoding, encryption, an arithmetic operation, and decryption
    with small integers.
9.  Confirm the BGV result is correct modulo the selected plaintext
    modulus.

Use the API provided by the installed OpenFHE version. Do not copy
method signatures blindly from old examples. Keep each smoke test small
and isolated.

### Phase D --- CKKS inference

Implement CKKS first, as originally planned.

1.  Select parameters using OpenFHE's documented security constraints
    and the actual circuit depth.
2.  Target a 128-bit security level where supported and justified by the
    chosen parameter set.
3.  Start with the smallest practical circuit and parameter set; do not
    assume a ring dimension or modulus size is secure merely because it
    appeared in an earlier note.
4.  Generate the public/secret key pair and the evaluation keys required
    by the operations actually used.
5.  Encode and encrypt a patient's 11-feature vector.
6.  Compute the weighted sum `z = X @ w + b` homomorphically.
7.  Implement the chosen polynomial approximation only after the linear
    score works and is validated.
8.  Decrypt the result on the client side.
9.  Compare the decrypted score with the corresponding plaintext score.
10. Run the same test procedure across the full held-out test set.
11. Measure key generation, encryption, evaluation, decryption, memory,
    ciphertext size, and key sizes where practical.

Be careful with packed-slot reductions and rotations. Generate the
required evaluation keys and verify the slot layout. Do not assume
`EvalSum` or a particular rotation-key call works without checking the
installed API.

### Phase E --- BGV readiness and inference

BGV operates over modular integer arithmetic. Quantization and modulus
selection are central parts of the experiment.

1.  Quantize the standardized input features and model weights using
    documented scale factors.
2.  Quantize the bias consistently with the product of the feature and
    weight scales.
3.  Estimate and then verify the maximum magnitude of every intermediate
    integer value.
4.  Choose a plaintext modulus that supports the required batching
    layout and prevents unintended modular wraparound for the tested
    circuit.
5.  Use a batching-compatible plaintext modulus where SIMD batching is
    required. Verify the congruence and library requirements for the
    selected ring dimension.
6.  Implement encrypted dot-product inference.
7.  Dequantize the result on the client.
8.  Compare it with the plaintext score and record classification
    agreement.
9.  Add polynomial evaluation only if the modulus and noise budget
    permit it.
10. If a cubic or quintic approximation is impractical, report that as
    an experimental limitation rather than hiding it. A linear decision
    score is a valid baseline, but compare like-for-like circuits where
    possible.

Do not treat a rough bit-count estimate as proof that modular wraparound
cannot occur. Include boundary tests and inspect decrypted
intermediate/final values.

### Phase F --- Fair comparison

Keep these conditions identical wherever the schemes allow:

-   dataset and exact record ordering;
-   train/test split and random seed;
-   feature preprocessing and feature order;
-   trained logistic-regression weights;
-   polynomial approximation and degree, when feasible;
-   security target;
-   hardware and operating environment;
-   CPU thread count;
-   timing methodology and warm-up policy.

Where the schemes require different parameters or cannot support the
same circuit, state the difference explicitly. Do not force identical
parameter values if that would make one scheme invalid or insecure.

Report at least:

-   plaintext accuracy, ROC AUC, F1, recall, and confusion matrix;
-   CKKS and BGV prediction agreement with plaintext;
-   CKKS score error (absolute and, where useful, mean/max error);
-   BGV quantization/dequantization error;
-   key-generation time;
-   encryption time;
-   homomorphic evaluation time;
-   decryption time;
-   peak memory;
-   ciphertext size;
-   public/evaluation key sizes;
-   parameter set, ring dimension, modulus information, and
    multiplicative depth;
-   throughput or latency per record.

Use repeated measurements where possible and report the number of
repetitions and summary statistic. Do not compare timings from different
machines as if they were controlled experiments.

### Phase G --- Client/server demonstration

After both inference paths work locally:

1.  Separate client-side key generation/encryption/decryption from
    server-side evaluation.
2.  Serialize ciphertexts and the necessary public/evaluation material
    using supported OpenFHE serialization APIs.
3.  Demonstrate that the server-side process has no secret key.
4.  Show that the server receives ciphertext data rather than raw
    patient features.
5.  Return an encrypted prediction/score and decrypt it on the client.
6.  Document the threat model and metadata leakage.

Do not build a production medical application. A command-line
demonstration is sufficient unless the user explicitly asks for a UI.

### Phase H --- Results and presentation material

Generate reproducible tables and plots from saved experiment outputs,
not hand-entered numbers.

Useful plots include:

-   latency by scheme and operation;
-   memory and serialized size by scheme;
-   score error / quantization error;
-   prediction agreement;
-   effect of polynomial degree;
-   effect of parameter choices, where the experiment is valid.

Include a limitations section. The dataset is small, public, and used as
a stand-in for patient-owned records. It does not establish clinical
utility or prove that a real hospital deployment is secure.

## 7. Experimental integrity and claims

Use precise language in code, comments, reports, and presentation
slides.

Acceptable claims, if supported by measurements:

-   The server can evaluate the implemented model without seeing the
    plaintext feature vector or holding the secret key.
-   CKKS supports approximate arithmetic and is naturally suited to
    real-valued model parameters.
-   BGV supports exact modular integer arithmetic, requiring
    quantization for this real-valued model.
-   The two schemes exhibit different accuracy, parameter, performance,
    and implementation trade-offs.

Do not claim:

-   encrypted inference is more accurate than plaintext inference;
-   CKKS or BGV is universally better;
-   all patient information is "fully protected" without describing the
    threat model and leakage;
-   a public dataset is itself private;
-   the system is clinically validated;
-   BGV's modular result is an unbounded exact integer result;
-   a benchmark is reproducible unless the environment and measurement
    procedure are recorded.

When comparing with Paillier or ordinary encryption, explain the
distinction accurately. Conventional encryption protects stored or
transmitted data but normally requires decryption before ordinary
computation. Paillier is additively homomorphic and does not directly
provide the general ciphertext-ciphertext multiplication needed for the
full polynomial model circuit. Do not imply that CKKS/BGV improve
predictive accuracy; their value is computation over encrypted data.

## 8. Code quality and repository organization

Use a simple, readable layout. Adapt to the existing repository instead
of reorganizing it without need.

A possible structure:

``` text
project/
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── artifacts/
│   ├── hf_plain.npz
│   └── hf_meta.json
├── src/
│   └── fhe_compare/
│       ├── __init__.py
│       ├── data.py
│       ├── train.py
│       ├── metrics.py
│       ├── ckks.py
│       ├── bgv.py
│       ├── client.py
│       └── server.py
├── scripts/
│   ├── train_plain.py
│   ├── ckks_smoke.py
│   └── bgv_smoke.py
├── tests/
└── results/
```

Guidelines:

-   Keep raw data, generated artifacts, and result files out of source
    modules.
-   Add suitable entries to `.gitignore` for virtual environments,
    caches, raw/private data, generated keys, ciphertexts, and large
    result artifacts.
-   Never commit secret keys or private patient data.
-   Keep cryptographic key material out of logs.
-   Use explicit types and docstrings for cryptographic interfaces.
-   Keep scheme-specific code separate, with shared preprocessing and
    evaluation utilities.
-   Store experiment configuration and results in machine-readable
    formats such as JSON or CSV.
-   Include a README with exact setup and run instructions.
-   Add tests for preprocessing, quantization, score calculations,
    serialization round trips, and decrypted-result correctness where
    practical.

## 9. Working protocol for Claude Code

At the start of a task:

1.  Summarize what you found in the repository.
2.  State the smallest next implementation step.
3.  Identify any assumptions or API uncertainties.
4.  Make the change.
5.  Run a targeted test or smoke test.
6.  Summarize changed files, commands run, results, and remaining
    issues.

Do not implement every phase in one response. Work one phase at a time
and wait for the user to review major milestones.

When blocked, provide: - the exact error; - the likely cause, separated
from confirmed facts; - the smallest diagnostic command; - the next safe
action.

The immediate next milestone is **audit the existing files and create a
working uv environment on Linux, then verify that OpenFHE imports and
both CKKS and BGV smoke tests can run**. Do not start the full encrypted
inference implementation until those checks pass.
