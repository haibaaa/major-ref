"""Phase G: client/server demonstration with file-based message exchange.

Each step runs as a separate OS process (no shared OpenFHE state):
    server init -> client setup -> client encrypt -> server evaluate -> client decrypt
During `server evaluate`, <run>/client_private is chmod 000, so the server
process cannot read the secret key even by mistake (not effective as root).

Afterwards the script audits everything the server could see (public/,
to_server/, from_server/):
  * the secret key's serialized bytes do not occur in any of those files;
  * the public key was not sent;
  * no preprocessed feature value or raw continuous value of any record
    occurs as float64 bytes in the request ciphertexts;
and compares the client's decrypted results with plaintext inference.

Usage:  uv run python scripts/demo_client_server.py [--runs ckks:1 bgv:1 ...]
Writes results/client_server_demo.json. Demo files go to demo/ (git-ignored).
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
SCRIPTS = ROOT / "scripts"
CONTINUOUS = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
              "serum_creatinine", "serum_sodium"]


def step(script: str, *args: str) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{script} {' '.join(args)} failed:\n{proc.stderr}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    out["process_wall_s"] = time.perf_counter() - t0
    return out


def server_visible(run: Path) -> list[Path]:
    return [p for d in ("public", "to_server", "from_server") for p in sorted((run / d).rglob("*")) if p.is_file()]


def contains_any(files: list[Path], needles: list[bytes]) -> list[str]:
    hits = []
    for f in files:
        data = f.read_bytes()
        hits += [f"{f.name}:{i}" for i, n in enumerate(needles) if n in data]
    return hits


def audit(run: Path) -> dict:
    files = server_visible(run)
    sk = (run / "client_private" / "secret_key.bin").read_bytes()
    pk = (run / "client_private" / "public_key.bin").read_bytes()
    # 64-byte windows from the key bodies (headers are skipped: they are generic)
    windows = lambda b: [b[o:o + 64] for o in range(len(b) // 4, len(b) - 64, len(b) // 8)]
    art = np.load(ROOT / "artifacts" / "hf_plain.npz")
    raw = pd.read_csv(ROOT / "data" / "Heart_failure_clinical_records_dataset.csv").iloc[art["idx_test"]]
    requests = sorted((run / "to_server" / "requests").glob("*.ct"))
    datas = [r.read_bytes() for r in requests]
    # (1) preprocessed values = what the client actually encodes; float64 bytes have
    #     ~52 bits of mantissa entropy, so any hit would indicate plaintext in the file.
    pre = sorted({struct.pack("<d", float(v)) for v in art["X_test"].ravel() if v != 0})
    # (2) raw continuous values are mostly round numbers (e.g. 60.0 = 6 zero bytes + 2),
    #     which also occur in serialization headers and by chance. Compare the hit rate in
    #     each record's own request with the rate in a different record's request.
    own = ctrl = n = 0
    for i, row in enumerate(raw[CONTINUOUS].to_numpy()):
        pats = {struct.pack("<d", float(v)) for v in row}
        own += sum(p in datas[i] for p in pats)
        ctrl += sum(p in datas[(i + 1) % len(datas)] for p in pats)
        n += len(pats)
    return {
        "server_visible_files": len(files),
        "secret_key_file_names_visible": [f.name for f in files if "secret" in f.name],
        "secret_key_byte_windows_checked": len(windows(sk)),
        "secret_key_byte_windows_found": contains_any(files, windows(sk)),
        "public_key_byte_windows_found_in_to_server": contains_any(
            [f for f in files if "to_server" in f.parts], windows(pk)),
        "preprocessed_feature_patterns_checked": len(pre),
        "preprocessed_feature_patterns_found_in_requests": len({p for p in pre if any(p in d for d in datas)}),
        "raw_value_hit_rate_own_request": own / n,
        "raw_value_hit_rate_other_record_request_control": ctrl / n,
        "note": "Byte audit only rules out accidental plaintext/secret-key inclusion; "
                "confidentiality itself rests on the RLWE hardness assumption.",
    }


def check_results(run: Path, scheme: str, degree: int | None) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from fhe_compare import bfv, bgv
    from fhe_compare.model import load_model

    m = load_model()
    dec = json.loads((run / "client_private" / "decrypted.json").read_text())
    z = np.array([r["z"] for r in dec])
    out = {"records": len(dec),
           "decision_agreement_with_plain": float(((z > 0) == (m.z_test > 0)).mean()),
           "z_abs_err_max": float(np.abs(z - m.z_test).max())}
    if degree:
        p = np.array([r["p"] for r in dec])
        out["p_abs_err_vs_plain_poly_max"] = float(np.abs(p - np.polynomial.polynomial.polyval(m.z_test, m.poly[degree])).max())
    if scheme in ("bgv", "bfv"):
        lib = bfv if scheme == "bfv" else bgv
        q = lib.Quantizer.build(m.w, m.b, m.clip, m.poly[1])
        exact = []
        for r, x in zip(dec, m.X_test):
            zq = q.exact_z(q.quantize_x(x))
            exact.append(r["raw_residues"][0] == zq and (not degree or r["raw_residues"][1] == q.exact_p(zq)))
        out["decrypted_equals_exact_integer"] = all(exact)
    return out


def sizes(run: Path) -> dict:
    ts = run / "to_server"
    size = lambda ps: sum(p.stat().st_size for p in ps)
    req = sorted((ts / "requests").glob("*.ct"))
    resp = sorted((run / "from_server").glob("*.ct"))
    return {"setup_bundle_bytes": {p.name: p.stat().st_size for p in sorted(ts.glob("*.bin"))},
            "setup_bundle_total_bytes": size(ts.glob("*.bin")),
            "request_bytes_per_record": size(req) / len(req),
            "response_bytes_per_record": size(resp) / len(req),
            "client_private_bytes": {p.name: p.stat().st_size for p in sorted((run / "client_private").iterdir())}}


def run_demo(scheme: str, degree: int | None) -> dict:
    run = DEMO / f"{scheme}_{'linear' if degree is None else f'deg{degree}'}"
    assert run.parent == DEMO
    if run.exists():
        shutil.rmtree(run)
    r = ["--run", str(run)]
    steps = {"server_init": step("fhe_server.py", "init", *r, "--scheme", scheme,
                                 *(["--degree", str(degree)] if degree else [])),
             "client_setup": step("fhe_client.py", "setup", *r),
             "client_encrypt": step("fhe_client.py", "encrypt", *r)}
    private = run / "client_private"
    private.chmod(0o000)
    try:
        steps["server_evaluate"] = step("fhe_server.py", "evaluate", *r)
    finally:
        private.chmod(0o700)
    steps["client_decrypt"] = step("fhe_client.py", "decrypt", *r)
    return {"run": run.name, "scheme": scheme, "degree": degree,
            "client_private_locked_during_server_step": os.geteuid() != 0,
            "steps": steps, "sizes": sizes(run), "audit": audit(run), "results": check_results(run, scheme, degree)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["ckks:1", "bgv:1"],
                    help="scheme:degree, degree 0 = linear score only")
    args = ap.parse_args()
    out = []
    for spec in args.runs:
        scheme, d = spec.split(":")
        degree = int(d) or None
        r = run_demo(scheme, degree)
        out.append(r)
        a, res = r["audit"], r["results"]
        print(f"{r['run']}: agreement={res['decision_agreement_with_plain']:.3f} "
              f"z_err_max={res['z_abs_err_max']:.1e}"
              + (f" exact={res['decrypted_equals_exact_integer']}" if scheme in ("bgv", "bfv") else "")
              + f" | setup bundle {r['sizes']['setup_bundle_total_bytes'] / 2**20:.1f} MiB, "
              f"request {r['sizes']['request_bytes_per_record'] / 1024:.0f} KiB, "
              f"response {r['sizes']['response_bytes_per_record'] / 1024:.0f} KiB"
              f" | sk windows found: {len(a['secret_key_byte_windows_found'])}/{a['secret_key_byte_windows_checked']}, "
              f"pk sent: {bool(a['public_key_byte_windows_found_in_to_server'])}, "
              f"preprocessed feature patterns found: {a['preprocessed_feature_patterns_found_in_requests']}/"
              f"{a['preprocessed_feature_patterns_checked']} | raw round-value hit rate own/control "
              f"{a['raw_value_hit_rate_own_request']:.3f}/{a['raw_value_hit_rate_other_record_request_control']:.3f}")
    path = ROOT / "results" / "client_server_demo.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
