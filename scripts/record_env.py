"""Record Python/uv/OpenFHE/OS/CPU details to results/environment.json."""

import json, os, platform, subprocess, sys
from importlib.metadata import version
from pathlib import Path


def cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "unavailable"


cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")), platform.processor())
env = {
    "python": sys.version.split()[0],
    "uv": cmd("uv", "--version"),
    "openfhe": version("openfhe"),
    "numpy": version("numpy"),
    "scikit-learn": version("scikit-learn"),
    "os": platform.freedesktop_os_release().get("PRETTY_NAME", platform.system()),
    "kernel": platform.release(),
    "glibc": "-".join(platform.libc_ver()),
    "machine": platform.machine(),
    "cpu": cpu,
    "logical_cpus": os.cpu_count(),
}
out = Path(__file__).resolve().parents[1] / "results" / "environment.json"
out.write_text(json.dumps(env, indent=2) + "\n")
print(json.dumps(env, indent=2))
