"""Pre-compute CPD features (severity nu and location gamma) for the full
STOXX 600 panel, parallelised across stocks.

Reads the processed long-format CSV produced by 01_data_loading.ipynb and writes
a CSV with columns:

    date, ticker, cpd_nu_<lbw>, cpd_gamma_<lbw>

For each ticker, we slide a window of length `lbw` along its 1d arithmetic
returns and call cpd_scores() at each step. With stride=1 this matches the
paper exactly, meaning the CPD algorithm is running the Gaussian Process 
optimization for every single trading day for every single ticker.

==============================================================================
USAGE
==============================================================================

All parameters can be set in configs/default.yaml under the `dmn:` section
(`cpd_lbw`, `cpd_stride`), or overridden via the command line. CLI flags take
precedence over YAML values.

Basic invocation (uses YAML defaults):
    python scripts/02_compute_cpd.py

Override LBW only:
    python scripts/02_compute_cpd.py --lbw 63

Override LBW and stride:
    python scripts/02_compute_cpd.py --lbw 21 --stride 1

Limit parallelism (e.g. to 4 cores):
    python scripts/02_compute_cpd.py --n-jobs 4

==============================================================================
WORKED EXAMPLES
==============================================================================

1) Paper-faithful precompute (LBW=21, stride=1, all cores) -- slow but exact:
       python scripts/02_compute_cpd.py --lbw 21 --stride 1

2) Fast iteration for development (LBW=21, stride=5):
       python scripts/02_compute_cpd.py --lbw 21 --stride 5

3) Multi-LBW sweep for a sensitivity analysis (run sequentially):
       python scripts/02_compute_cpd.py --lbw 10  --stride 5
       python scripts/02_compute_cpd.py --lbw 21  --stride 5
       python scripts/02_compute_cpd.py --lbw 63  --stride 5
       python scripts/02_compute_cpd.py --lbw 126 --stride 5

==============================================================================
OUTPUTS
==============================================================================

Saved to data/processed/cpd/ with self-documenting filenames:
    cpd_features_lbw<LBW>_s<STRIDE>.csv

Examples:
    cpd_features_lbw21_s5.csv     stride 5, LBW 21
    cpd_features_lbw21_s1.csv     stride 1, LBW 21 
    cpd_features_lbw63_s5.csv     stride 5, LBW 63
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cpd import cpd_scores


def compute_for_ticker(
    ticker: str,
    returns: np.ndarray,
    dates: pd.Series,
    lbw: int,
    stride: int,
) -> pd.DataFrame:
    """Compute (nu, gamma) for every valid window endpoint of a single ticker."""
    n = len(returns)
    nu = np.full(n, np.nan)
    gamma = np.full(n, np.nan)

    for t in range(lbw, n, stride):
        window = returns[t - lbw : t]
        if np.isnan(window).any():
            continue
        try:
            nu_t, gamma_t = cpd_scores(window, lbw)
            nu[t] = nu_t
            gamma[t] = gamma_t
        except Exception:
            # Optimisation failures → leave as NaN; downstream code can ffill
            continue

    return pd.DataFrame({
        "date":   dates.values,
        "ticker": ticker,
        f"cpd_nu_{lbw}":    nu,
        f"cpd_gamma_{lbw}": gamma,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute CPD features for the STOXX 600 panel."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lbw", type=int, default=None,
                        help="Lookback window for the GP CPD module "
                             "(default: cfg.dmn.cpd_lbw from YAML)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between consecutive CPD windows "
                             "(default: cfg.dmn.cpd_stride from YAML)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI args override YAML; YAML provides the defaults
    dmn_cfg = cfg.get("dmn", {})
    lbw    = args.lbw    if args.lbw    is not None else dmn_cfg.get("cpd_lbw", 21)
    stride = args.stride if args.stride is not None else dmn_cfg.get("cpd_stride", 1)
    # If you don't pass --stride in the terminal, the script goes to dmn_cfg  and looks for cpd_stride (5)

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    df = pd.read_csv(
        processed_dir / "stoxx600_processed.csv",
        parse_dates=["date"],
        usecols=["date", "ticker", "1d_arith_ret"],
    )
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = df["ticker"].unique()
    print(f"Computing CPD (lbw={lbw}, stride={stride}) on {len(tickers)} tickers, "
          f"{args.n_jobs} jobs")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(compute_for_ticker)(
            ticker,
            df.loc[df["ticker"] == ticker, "1d_arith_ret"].values,
            df.loc[df["ticker"] == ticker, "date"],
            lbw,
            stride,
        )
        for ticker in tickers
    )
    elapsed = time.time() - t0

    out = pd.concat(results, ignore_index=True)
    cpd_dir = PROJECT_ROOT / cfg["data"]["processed_cpd"]
    cpd_dir.mkdir(parents=True, exist_ok=True)
    out_path = cpd_dir / f"cpd_features_lbw{lbw}_s{stride}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Coverage: {out[f'cpd_nu_{lbw}'].notna().mean():.1%} of cells")


if __name__ == "__main__":
    main()