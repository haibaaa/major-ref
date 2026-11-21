# OpenFHE notes (verified on `openfhe==1.6.1.0.24.4`, Python 3.12)

Everything here was observed in this repository by running code, unless marked
otherwise. Re-verify after any OpenFHE upgrade — several items are
version-specific behaviour, not documented guarantees.

## 1. Installation

- PyPI wheels are tagged `py3-none-any` but contain
  `openfhe/openfhe.cpython-312-x86_64-linux-gnu.so` plus bundled
  `libOPENFHE{core,pke,binfhe}.so.1` and `libgomp.so`. Metadata says
  `Requires-Python: >=3.12`; in practice **only 3.12 can import it**.
- Version suffix `.22.4` / `.24.4` = build toolchain (Ubuntu 22.04 / 24.04). The
  `.24.4` build works on Arch (glibc 2.44).
- A successful install is not proof of a working extension; always import and
  run a smoke test.

## 2. API surface used (names checked with `dir()` / docstrings)

- Parameters: `CCParamsCKKSRNS`, `CCParamsBGVRNS` with `SetMultiplicativeDepth`,
  `SetScalingModSize`, `SetFirstModSize`, `SetBatchSize`, `SetPlaintextModulus`,
  `SetSecurityLevel(SecurityLevel.HEStd_128_classic)`, `SetScalingTechnique`,
  `SetKeySwitchTechnique`, `SetRingDim`.
- `GenCryptoContext(params)`; `cc.Enable(PKESchemeFeature.X)` for PKE, KEYSWITCH,
  LEVELEDSHE, ADVANCEDSHE.
- Keys: `cc.KeyGen()` → `.publicKey`, `.secretKey`; `EvalMultKeyGen`,
  `EvalSumKeyGen`, `EvalRotateKeyGen(sk, [indices])`.
- Encoding: `MakeCKKSPackedPlaintext(np.ndarray)` (fast path for float64 arrays),
  `MakePackedPlaintext(list[int])`.
- Ops: `EvalMult` (ct×ct and ct×pt overloads), `EvalAdd` (ct+ct, ct+pt),
  `EvalSum(ct, batchSize)`, `EvalRotate(ct, k)`, `EvalPoly(ct, coeffs)`,
  `ModReduce` (needed with BGV FIXEDMANUAL).
- Decrypt: `cc.Decrypt(ct, sk)` → `Plaintext`; `pt.SetLength(n)`;
  `GetRealPackedValue()` (CKKS), `GetPackedValue()` (BGV, centered residues).
- Introspection: `GetRingDimension()`, `GetPlaintextModulus()`,
  `GetKeySwitchTechnique()`, `GetExistingEvalAutomorphismKeyIndices(keyTag)`.
- Serialization: module-level `Serialize(obj, BINARY)` → bytes;
  `DeserializeCryptoContextString`, `DeserializeCiphertextString`,
  `DeserializePrivateKeyString`, `DeserializePublicKeyString` (bytes, BINARY);
  `SerializeEvalMultKeyString(BINARY)`, `SerializeEvalAutomorphismKeyString(BINARY)`
  and the matching `Deserialize…String` functions, which **insert keys into a
  library-global cache**. File variants exist: `cc.SerializeEvalMultKey(path, BINARY)`,
  `cc.SerializeEvalSumKey`, `cc.SerializeEvalAutomorphismKey`.
- Global cache reset: `CryptoContext.ClearStaticMapsAndVectors()` (docstring:
  clears evaluation-key maps and the context registry);
  `cc.ClearEvalAutomorphismKeys()`. There is **no** `ClearEvalMultKeys` binding.

## 3. Behaviour and pitfalls (measured)

### Features and parameters

1. `EvalSumKeyGen` raises
   `EvalSumKeyGen operation has not been enabled. Enable(ADVANCEDSHE) must be called`
   unless `ADVANCEDSHE` is enabled. `EvalPoly` is also an advanced-SHE operation.
2. At `HEStd_128_classic`, OpenFHE chose **N = 16384** for every configuration
   here (CKKS depth 1–4, BGV depth 1–3 with t up to 33 bits). Forcing
   `SetRingDim(8192)` for CKKS depth 1 / scale 50 / first mod 60 fails with
   `The specified ring dimension (8192) does not comply with HE standards
   recommendation (16384)`. Likely reason (unverified): the auxiliary
   key-switching moduli of HYBRID key switching push log(QP) past the 8192 bound.
3. Default key-switching technique reported: `HYBRID`.

### CKKS

4. Minimum depth found by trial (`scripts/ckks_poly.py`): degree 1 → 2,
   degree 3 → 4, degree 5 → 4 (circuit includes the depth-1 ct×pt weight
   product). Errors seen at lower depth:
   - `Decode(): The decryption failed because the approximation error is too high`
   - `DropLastElement(): Removing last element of DCRTPoly renders it invalid`
   - `SetLevel(): The current multiplicative depth [2] is insufficient; increase it`
5. `EvalPoly` has an int-coefficient overload listed first and a float overload;
   passing a list of Python floats selects the float overload (results match the
   plaintext polynomial to ~1e-13, so no truncation).
6. `EvalPoly` degree 1 works **without** a relinearization key.
7. Observed numerical error: ~1e-13 per operation chain at scale 50 bits;
   degree 5 up to ~3e-11.

### BGV

8. **`EvalSum` gives suffix sums in BGV**: for input [1..11, 0×5] with batch 16,
   output was `[66, 65, 63, 60, 56, 51, 45, 38, 30, 21, 11, 0, 0, 0, 0, 0]`.
   CKKS with `SetBatchSize(16)` put the total in every slot. Read slot 0 only.
9. Plaintext modulus must be prime with `t ≡ 1 (mod 2N)` for batching; we use
   `t ≡ 1 (mod 2^16)`, valid for N ≤ 2^15. `GetPlaintextModulus()` returns t
   unchanged.
10. **Plaintext modulus size limit** (`scripts/bgv_modulus_probe.py`, depth 3, 3 repeats):
    - default FLEXIBLEAUTO: 30, 31, 33 bits correct; 32 bits →
      `PrecomputeCRTTables(): EstimateLogP() failure: expected sizeP [2], but got sizeP [3]`
      (depends on the particular prime); ≥ 34 bits →
      `Reduce the number of additions or key switches per level or the digit size.
      We do not support moduli greater than 60 bits.`
    - FIXEDMANUAL with `SetScalingModSize(60)`, `SetFirstModSize(60)`: 30–32 bits
      correct; **33, 34, 36, 40 bits build a context and decrypt WRONG values with
      no error (0/3 correct)**. Without explicit 60-bit moduli, FIXEDMANUAL at
      40 and 45 bits also built and gave wrong results.
    - FLEXIBLEAUTO with `KeySwitchTechnique.BV`: fails with
      `digitSize is not allowed to be 0 for BV key switching in BGV when scalingModSize = 0`.
    - Lesson: never accept a BGV parameter set without comparing decrypted
      values to exact integer arithmetic, including values near ±t/2.
11. Decrypted values are centered residues; wraparound is silent
    (smoke test: 300·300 = 90000 decrypts as 24463 with t = 65537).

### Rotations and evaluation keys

12. `EvalRotate(ct, k)`: positive k = left shift (slot i receives slot i+k).
13. Key counts: `EvalSumKeyGen` (batch 16) → 6 automorphism keys (indices
    5, 25, 125, 625, 19025, 30177 for N = 16384); `EvalRotateKeyGen([1,2,4,8])`
    → 4 (5, 25, 625, 30177). Each ≈ 2.6 MiB at depth 1.
14. `SerializeEvalSumKey` and `SerializeEvalAutomorphismKey` produced identical
    bytes — EvalSum keys live in the automorphism-key map.
15. Speed: `EvalSum` ≈ 22 ms vs rotate-and-add ≈ 34 ms (100 iterations, 1 thread,
    depth 1, both schemes). Cause not investigated.

### Serialization and global state

16. Evaluation keys are stored in **library-global maps** keyed by secret-key
    tag. Consequences:
    - in one process, repeated keygen accumulates keys: peak RSS rose from
      ~231 MB to ~400 MB over 5 benchmark repetitions until
      `ClearStaticMapsAndVectors()` was called after each repetition;
    - deserializing keys inserts them into this cache (they are found via the
      deserialized context's tag);
    - for a true client/server separation use separate processes.
17. `cc.SerializeEvalMultKey(path, BINARY)` returns True and writes a **9-byte
    empty map** when no relinearization key exists. `sizes.eval_key_sizes`
    reports 0 for key sets that were not generated.
18. A full round trip (context, eval keys, ciphertexts, secret key) after
    `ClearStaticMapsAndVectors()` reproduces exact BGV results and CKKS results
    within 1e-6 (test `test_serialization.py`).
19. Serialized header contains parameter doubles (e.g. 1.0 and 30.0 were found
    near byte 500–570); naive byte-pattern searches for round feature values
    therefore produce false positives (see `experiments.md` G-2).

### Sizes at N = 16384 (BINARY)

Ciphertext ≈ 256 KiB per RNS limb (2 polynomials × 16384 coefficients × 8 bytes),
consistent with depth + 2 limbs fresh: depth 1 → 769 KiB, depth 2 → 1025 KiB,
depth 4 → 1537 KiB (the limb count is inferred from sizes, not read from the API). Outputs are
smaller after mod-reduction (e.g. degree-1 z 769 KiB, p 513 KiB). Context
serialization alone is < 1 KiB (no keys inside).

## 4. Threads

OpenFHE uses OpenMP (bundled libgomp). Set `OMP_NUM_THREADS` **before** the
process imports openfhe (the benchmark passes it to worker subprocesses). With 4
threads evaluation is ~1.5× faster; encryption did not speed up.
