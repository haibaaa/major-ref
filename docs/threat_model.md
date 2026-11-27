# Threat model and limitations of the client/server demonstration

This describes what `scripts/demo_client_server.py` (and the `fhe_compare.client` /
`fhe_compare.server` modules) do and do not protect for CKKS, BGV and BFV. It is a technical
demonstration on a public dataset, not a clinical or production system.

## Parties and what each holds

| | Client (patient / data owner) | Server (model owner / evaluator) |
|---|---|---|
| Raw and preprocessed patient features | yes | **no** — receives ciphertexts only |
| Secret key | yes (`client_private/secret_key.bin`, mode 600) | **no** |
| Public (encryption) key | yes | not sent (the circuit uses only ciphertext–plaintext operations) |
| Crypto context, evaluation keys (rotation, relinearization) | generates them | yes — required to evaluate |
| Model weights, bias, polynomial coefficients, BGV scales | no (only the scales needed to dequantize) | yes, in plaintext |
| Preprocessing statistics (train means/scales, clip) | yes, published in `protocol.json` | yes |
| Decrypted score `z`, polynomial score `p`, decision | yes | **no** |

## What the demonstration checks (measured, `results/client_server_demo.json`)

- Every role runs as a separate OS process; the only exchange is files.
- During server evaluation `client_private/` is `chmod 000`, so the server
  process cannot read the secret key (this does not hold for a root user).
- Byte audit of every server-visible file: no 64-byte window of the serialized
  secret key occurs; the public key was not sent; none of the 180 distinct
  preprocessed feature values (float64) of the 60 test records occurs in the
  requests. Raw integer-like values (e.g. age 60.0) occur at the same rate in a
  record's own request as in another record's request (control), i.e. by chance
  and in serialization headers, not because of the data.
- Decrypted decisions agree with plaintext inference on all 60 test records;
  BGV results equal the exact integer computation.

The byte audit rules out *accidental* inclusion of plaintext or the secret key
in the messages. It is not a proof of confidentiality.

## What is protected, and under which assumption

Against an **honest-but-curious server** (follows the protocol, tries to learn
from what it sees), the feature vectors and the returned scores are
confidential, assuming the hardness of Ring-LWE for the parameter sets OpenFHE
selected under `HEStd_128_classic` (ring dimension 16384 for every circuit
used; OpenFHE rejected 8192 for these settings as non-compliant with the HE
standard recommendations).

## What is not protected or not addressed

- **Metadata.** The server sees how many records are sent, when, message sizes
  (fixed per circuit, e.g. ~1 MiB per request for the degree-1 circuit) and
  its own evaluation timing. Our timings were not analysed for data dependence.
- **Model confidentiality.** The weights are plaintext on the server. Conversely,
  the client decrypts the linear score `z`; with 12 linearly independent queries
  it could solve for the 11 weights and the bias exactly (BGV) or to high
  precision (CKKS). Model-extraction protection is out of scope.
- **Result integrity.** Homomorphic encryption does not prove that the server
  evaluated the agreed model. A malicious server could return any ciphertext.
- **Sharing decrypted CKKS results.** CKKS decryptions contain noise that depends
  on the secret key; Li and Micciancio (EUROCRYPT 2021) showed that revealing
  decrypted CKKS values to an adversary can enable key recovery. The client
  must not send decrypted values back to the server.
- **Malicious clients.** The server cannot see whether inputs were clipped to
  [-4, 4]. Out-of-range inputs can push BGV values past the wraparound bound or
  CKKS scores outside the polynomial fit range; this corrupts only that
  client's own result.
- **Authentication, authorization, transport security, secure key storage.**
  Not implemented: files on one machine, no TLS, the secret key is an ordinary
  file protected only by filesystem permissions.
- **Outputs are not calibrated clinical probabilities.** The decision is taken
  from `sign(z)`; `p` is a polynomial score valid only inside the fit range
  (outside it, degree 3 inverts the decision for |z| ≥ 6.02).
- **Data.** The Heart Failure Clinical Records dataset is public and is used as
  a stand-in for patient-owned records. Nothing here establishes clinical
  utility or the security of a real deployment.
