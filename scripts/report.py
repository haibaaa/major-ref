"""Build Markdown comparison tables from saved benchmark JSON files.

Usage:  uv run python scripts/report.py [results/benchmark_threads1_evalsum.json ...]
Default: every results/benchmark_*.json. Writes results/benchmark_tables.md.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIB, MIB = 1024, 1024 * 1024


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section(path: Path) -> str:
    d = json.loads(path.read_text())
    env = d["environment"]
    cfgs = d["configs"]
    lines = [f"## {path.name}",
             f"CPU: {env['cpu']} | OS: {env['os']} | Python {env['python']} | OpenFHE {env['openfhe']} | "
             f"OMP_NUM_THREADS={d['omp_num_threads']} | slot sum: {d['slot_sum']} | "
             f"repetitions={d['repetitions']}, warm-up records/rep={d['warmup_records_per_rep']} | "
             f"statistic: {d['summary_statistic']}", ""]

    lines.append(table(
        ["config", "N", "depth", "outputs", "keygen ms", "encrypt ms", "evaluate ms (p95)", "decrypt ms",
         "latency ms", "records/s", "peak RSS MB (1st rep / all)"],
        [[c["name"], str(c["params"]["ring_dim"]), str(c["params"]["mult_depth"]), c["outputs"],
          f"{c['timing']['keygen']['median_ms']:.1f}", f"{c['timing']['encrypt']['median_ms']:.2f}",
          f"{c['timing']['evaluate']['median_ms']:.2f} ({c['timing']['evaluate']['p95_ms']:.2f})",
          f"{c['timing']['decrypt']['median_ms']:.2f}", f"{c['timing']['latency_per_record']['median_ms']:.2f}",
          f"{c['timing']['throughput_records_per_s_sequential']:.1f}", f"{c['memory']['peak_rss_mb_first_rep']:.0f} / {c['memory']['peak_rss_mb_all_reps']:.0f}"]
         for c in cfgs]))
    lines.append("")

    lines.append(table(
        ["config", "public key MiB", "relin key MiB", "rotation keys MiB", "input ct KiB", "output ct(s) KiB",
         "secret key KiB (client only)"],
        [[c["name"], f"{c['sizes_bytes']['public_key'] / MIB:.2f}", f"{c['sizes_bytes']['relin_key'] / MIB:.2f}",
          f"{c['sizes_bytes']['rotation_keys'] / MIB:.2f}", f"{c['sizes_bytes']['input_ciphertext'] / KIB:.0f}",
          " + ".join(f"{s / KIB:.0f}" for s in c["sizes_bytes"]["output_ciphertexts"]),
          f"{c['sizes_bytes']['secret_key_client_only'] / KIB:.0f}"] for c in cfgs]))
    lines.append("")

    rows = []
    for c in cfgs:
        k = c["correctness"]
        if c["scheme"] == "ckks":
            z_err = f"{k['z_abs_err_vs_plain']['max']:.1e} (encryption)"
            p_err = f"{k['p_abs_err_vs_plain_poly']['max']:.1e}" if "p_abs_err_vs_plain_poly" in k else "-"
            exact = "n/a (approximate)"
        else:
            z_err = f"{k['z_quant_err_vs_plain']['max']:.1e} (quantization)"
            p_err = f"{k['p_quant_err_vs_plain_poly']['max']:.1e}" if "p_quant_err_vs_plain_poly" in k else "-"
            exact = f"{k['decrypted_equals_exact_integer']} / edge {k['edge_decrypted_equals_exact_integer']}"
        sig = f"{k['p_abs_err_vs_sigmoid']['max']:.3f}" if "p_abs_err_vs_sigmoid" in k else "-"
        m = c["test_metrics_from_decrypted"]
        rows.append([c["name"], z_err, p_err, sig, exact, f"{k['decision_agreement_with_plain']:.3f}",
                     f"{m['accuracy']:.3f}", f"{m['roc_auc']:.3f}", f"{m['f1']:.3f}", f"{m['recall']:.3f}",
                     str(m["confusion_matrix"])])
    lines.append(table(["config", "max z err", "max p err vs plain poly", "max p err vs sigmoid",
                        "exact integer (test / edge)", "agreement", "acc", "AUC", "F1", "recall", "CM [[TN,FP],[FN,TP]]"],
                       rows))
    return "\n".join(lines) + "\n"


def paillier_section(path: Path) -> str:
    d = json.loads(path.read_text())
    t, c, sizes = d["timing"], d["correctness"], d["approx_decimal_wire_bytes"]
    return "\n".join([
        f"## {path.name} — Paillier additive baseline",
        "Linear score only; each of the 11 features is encrypted separately. "
        "Decimal wire sizes are estimates and are not directly comparable to OpenFHE binary serialization.",
        f"Security note: {d['security_note']}",
        f"Sample: {d['records_per_repetition']} test records × {d['repetitions']} repetition(s); {d.get('record_sampling', 'full fixed test set')}.",
        "",
        table(["key bits", "keygen ms", "encrypt record ms", "evaluate ms", "decrypt ms",
               "max score error", "decision agreement", "request bytes est.", "response bytes est."],
              [[str(d["key_bits"]), f"{t['keygen']['median_ms']:.1f}",
                f"{t['encrypt_record']['median_ms']:.2f}", f"{t['evaluate_record']['median_ms']:.2f}",
                f"{t['decrypt_score']['median_ms']:.2f}", f"{c['max_abs_score_error']:.2e}",
                f"{c['decision_agreement']:.3f}", str(int(sizes["request_per_record"])),
                str(int(sizes["response_per_record"]))]]),
        ""])


def margin_section(path: Path) -> str:
    d = json.loads(path.read_text())
    rows = []
    for c in d["configurations"]:
        cfg, s = c["configuration"], c["summary"]
        if "sx_bits" in cfg:
            setting = f"t={cfg['t_bits']} sx=2^{cfg['sx_bits']} sw=2^{cfg['sw_bits']}"
        elif "modulus_bits" in cfg:
            setting = f"Paillier modulus={cfg['modulus_bits']} bits"
        else:
            setting = f"CKKS scale={cfg['scale_bits']} first={cfg['first_mod_bits']} bits"
        rows.append([
            c["scheme"],
            setting,
            f"{cfg['score_error_bound']:.2e}" if "score_error_bound" in cfg else "—",
            str(cfg.get("bound_below_train_min_margin", "—")),
            str(cfg.get("training_decision_agreement", "—")),
            str(cfg.get("test_decision_agreement", "—")),
            f"{s['decision_flips']}/{s['labeled_record_count']}",
            f"{s['max_abs_score_error']:.2e}",
        ])
    return "\n".join([
        f"## {path.name} — controlled margin stress",
        f"{len(d['target_scores'])} target logits; {d['configurations'][0]['summary']['record_count']} inputs per CKKS/BGV/BFV configuration, "
        f"{d['configurations'][-1]['summary']['record_count']} aligned inputs for Paillier. Exact zero is treated as a tie.",
        "",
        table(["scheme", "configuration", "score error bound", "bound < train margin",
               "train agreement", "test agreement", "synthetic flips", "max synthetic error"], rows),
        "",
    ])


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "results").glob("benchmark_threads*.json"))
    md = "# Benchmark tables (generated by scripts/report.py)\n\n" + "\n".join(section(p) for p in paths)
    if not sys.argv[1:]:
        for paillier_path in sorted((ROOT / "results").glob("benchmark_paillier*.json")):
            md += "\n" + paillier_section(paillier_path)
        margin_path = ROOT / "results" / "margin_stress.json"
        if margin_path.exists():
            md += "\n" + margin_section(margin_path)
    out = ROOT / "results" / "benchmark_tables.md"
    out.write_text(md)
    print(md)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
