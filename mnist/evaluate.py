"""Evaluation entry point: aggregate saved metrics or evaluate a checkpoint."""

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "utils"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "checkpoint":
        sys.argv = [str(TOOLS / "evaluate_checkpoint.py"), *sys.argv[2:]]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["report", "plot", "auc", "tssl", "checkpoint"]
    )
    parser.add_argument(
        "model", nargs="?", help="result directory for auc (e.g. logistic_kair)"
    )
    args = parser.parse_args()
    scripts = {
        "report": "build_comparison.py",
        "plot": "plot_results.py",
        "auc": "evaluate_auc.py",
        "tssl": "evaluate_tssl.py",
    }
    if args.action == "auc" and not args.model:
        parser.error("auc requires a result directory")
    sys.path.insert(0, str(TOOLS))
    sys.argv = [str(TOOLS / scripts[args.action])] + (
        [args.model] if args.model else []
    )
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
