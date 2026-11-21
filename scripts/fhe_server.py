"""SERVER command line.

    uv run python scripts/fhe_server.py init     --run demo/ckks_deg1 --scheme ckks --degree 1
    uv run python scripts/fhe_server.py evaluate --run demo/ckks_deg1

`evaluate` reads only <run>/public and <run>/to_server and writes <run>/from_server.
"""

import argparse
import json
from pathlib import Path

from fhe_compare import server
from fhe_compare.protocol import Layout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["init", "evaluate"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scheme", choices=["ckks", "bgv"])
    ap.add_argument("--degree", type=int)
    ap.add_argument("--sum", default="evalsum", choices=["evalsum", "rotations"])
    args = ap.parse_args()
    layout = Layout(args.run)
    if args.step == "init":
        proto = server.init(layout, args.scheme, args.degree, args.sum)
        print(json.dumps({"protocol": layout.protocol.name, "scheme": proto.scheme, "degree": proto.degree}))
    else:
        print(json.dumps(server.evaluate(layout)))


if __name__ == "__main__":
    main()
