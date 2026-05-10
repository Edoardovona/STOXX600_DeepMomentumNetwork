"""Train all walk-forward folds in sequence and persist results.

Each fold is trained by importing and calling train_dmn() from 03_train_dmn.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Module name starts with a digit, so we have to import it dynamically
_spec = importlib.util.spec_from_file_location(
    "train_dmn_mod", PROJECT_ROOT / "scripts" / "03_train_dmn.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_dmn = _mod.train_dmn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward orchestrator for the Deep Momentum Network."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="all",
                        help="Comma-separated fold indices, or 'all'")
    parser.add_argument("--fold-type", choices=["expanding", "rolling"], default=None,
                        help="Override the fold_type set in the YAML config")
    parser.add_argument("--use-cpd", dest="use_cpd", action="store_true")
    parser.add_argument("--no-cpd", dest="use_cpd", action="store_false")
    parser.set_defaults(use_cpd=True)
    return parser.parse_args()


def resolve_fold_indices(spec: str, n_folds: int) -> list[int]:
    if spec == "all":
        return list(range(n_folds))
    return [int(i) for i in spec.split(",")]


def main() -> None:
    args = parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fold_type:
        cfg["fold_type"] = args.fold_type

    fold_type = cfg.get("fold_type", "expanding")
    folds = cfg[f"folds_{fold_type}"]
    fold_indices = resolve_fold_indices(args.folds, len(folds))

    print(f"Walk-forward ({fold_type}, use_cpd={args.use_cpd}): "
          f"training folds {fold_indices}")

    results = []
    for i in fold_indices:
        ckpt = train_dmn(fold_idx=i, cfg=cfg, use_cpd=args.use_cpd, verbose=True)
        results.append((i, ckpt["test_metrics"]["sharpe"]))

    print("\nSummary:")
    for i, sharpe in results:
        print(f"  Fold {i}: out-of-sample Sharpe = {sharpe:+.3f}")


if __name__ == "__main__":
    main()