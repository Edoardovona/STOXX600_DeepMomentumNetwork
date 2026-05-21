"""Train all walk-forward folds in sequence and persist results.

This is the orchestrator that imports and calls train_dmn() from
scripts/03_train_dmn.py for each fold. Use this for the full pipeline run;
use 03_train_dmn.py directly for single-fold debugging.

==============================================================================
USAGE
==============================================================================

All flags accepted by 03_train_dmn.py are also accepted here, plus walk-forward
controls. CLI flags override YAML values.

Walk-forward controls:
    --folds 0,1,2        Comma-separated fold indices (default: all)
    --fold-type rolling  Override fold_type in YAML (expanding or rolling)
    --config PATH        Use a different YAML config (default: configs/default.yaml)

Model variants (same as 03_train_dmn.py):
    --use-cpd / --no-cpd
    --long-only / --no-long-only
    --tc 0.0025

==============================================================================
WORKED EXAMPLES
==============================================================================

1) Train all 4 expanding-window folds, paper baseline (no CPD):
       python scripts/03bis_walk_forward.py --no-cpd --no-long-only

2) Paper main result (with CPD), all folds:
       python scripts/03bis_walk_forward.py --use-cpd --no-long-only

3) Long-only adaptation, with CPD, all folds:
       python scripts/03bis_walk_forward.py --use-cpd --long-only

4) Realistic deployment (long-only + 25 bps costs), all folds:
       python scripts/03bis_walk_forward.py --use-cpd --long-only --tc 0.0025

5) Rolling-window sensitivity check, with CPD:
       python scripts/03bis_walk_forward.py --use-cpd --fold-type rolling

6) Re-run only specific folds (e.g. for debugging fold 2):
       python scripts/03bis_walk_forward.py --use-cpd --folds 2

7) Re-run multiple specific folds:
       python scripts/03bis_walk_forward.py --use-cpd --folds 0,1,3

==============================================================================
OUTPUTS
==============================================================================

For each fold, two files are written to data/processed/models/:
    dmn_fold<i>_<suffix>.pt
    predictions_fold<i>_<suffix>.csv

The orchestrator prints a summary table of per-fold Sharpe ratios at the end.
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
    parser.add_argument("--use-cpd", dest="use_cpd", action="store_true", default=None)
    parser.add_argument("--no-cpd",  dest="use_cpd", action="store_false")
    parser.add_argument("--long-only",    dest="long_only", action="store_true", default=None)
    parser.add_argument("--no-long-only", dest="long_only", action="store_false")
    parser.add_argument("--tc", dest="transaction_cost", type=float, default=None,
                        help="Transaction cost (decimal, e.g. 0.0025 for 25 bps)")
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

    print(f"Walk-forward ({fold_type}, use_cpd={args.use_cpd}, "
          f"long_only={args.long_only}, tc={args.transaction_cost}): "
          f"training folds {fold_indices}")

    results = []
    for i in fold_indices:
        ckpt = train_dmn(
            fold_idx=i,
            cfg=cfg,
            use_cpd=args.use_cpd,
            long_only=args.long_only,
            transaction_cost=args.transaction_cost,
            verbose=True,
        )
        results.append((i, ckpt["test_metrics"]["sharpe"]))

    print("\nSummary:")
    for i, sharpe in results:
        print(f"  Fold {i}: out-of-sample Sharpe = {sharpe:+.3f}")


if __name__ == "__main__":
    main()