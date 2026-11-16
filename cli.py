"""CLI: generate / train / monitor / retrain / check / demo.

check exit codes: 0 = no trigger (healthy), 1 = trigger fired,
2 = data/IO error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import generate_all  # noqa: E402
from lifecycle import run_lifecycle  # noqa: E402
from registry import default_registry_path, save_registry  # noqa: E402

EXIT_OK = 0
EXIT_TRIGGER = 1
EXIT_ERROR = 2


def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv=None):
    p = argparse.ArgumentParser(prog="cli.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    sub.add_parser("train")
    m = sub.add_parser("monitor"); m.add_argument("--scenario", default="drift", choices=["drift", "control"])
    r = sub.add_parser("retrain"); r.add_argument("--scenario", default="drift", choices=["drift", "control"])
    c = sub.add_parser("check"); c.add_argument("--scenario", default="drift", choices=["drift", "control"])
    c.add_argument("--registry", default=None)
    sub.add_parser("demo")
    args = p.parse_args(argv)

    try:
        if args.cmd == "generate":
            data = generate_all()
            _dump({k: {"n": len(v),
                       "label_rate": round(sum(r["label"] for r in v) / len(v), 6)}
                   for k, v in data.items()})
            return EXIT_OK
        if args.cmd == "demo":
            import demo as demo_mod
            return demo_mod.run()
        # train / monitor / retrain / check all run the lifecycle
        scenario = getattr(args, "scenario", "drift")
        report, reg = run_lifecycle(generate_all(), scenario=scenario)
        if args.cmd == "train":
            _dump({"champion": report["champion"],
                   "versions": [v["version"] for v in reg["versions"]]})
            return EXIT_OK
        if args.cmd == "monitor":
            _dump(report["monitoring"])
            return EXIT_OK
        if args.cmd == "retrain":
            save_registry(reg, default_registry_path())
            _dump(report)
            return EXIT_OK
        if args.cmd == "check":
            if args.registry and not os.path.exists(args.registry):
                print(f"error: registry not found: {args.registry}", file=sys.stderr)
                return EXIT_ERROR
            _dump(report["monitoring"])
            return EXIT_TRIGGER if report["monitoring"]["trigger_fired"] else EXIT_OK
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
