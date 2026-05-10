"""Pre-compute CPD features (severity nu and location gamma) for the full STOXX 600 panel, parallelised across stocks.

Reads the processed long-format CSV produced by 01_data_loading.ipynb and writes a csv file with columns:
    date, ticker, cpd_nu_<lbw>, cpd_gamma_<lbw>

For each ticker, we slide a window of length `lbw` along its 1d arithmetic returns and call cpd_scores() at each step.
 With stride=1 this matches the paper exactly; we expose stride as a parameter for cheap experimentation.

Usage:
    python scripts/02_compute_cpd.py --lbw 21 --stride 5 --n-jobs -1
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
            # Optimisation failures -> leave as NaN; downstream code can ffill
            continue

    return pd.DataFrame({
        "date": dates.values,
        "ticker": ticker,
        f"cpd_nu_{lbw}": nu,
        f"cpd_gamma_{lbw}": gamma,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lbw", type=int, default=21,
                        help="Lookback window for the GP CPD module")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between consecutive CPD windows")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    df = pd.read_csv(
        processed_dir / "stoxx600_processed.csv",
        parse_dates=["date"],
        usecols=["date", "ticker", "1d_arith_ret"],
    )
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = df["ticker"].unique()
    print(f"Computing CPD (lbw={args.lbw}, stride={args.stride}) on {len(tickers)} tickers, {args.n_jobs} jobs")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(compute_for_ticker)(
            ticker,
            df.loc[df["ticker"] == ticker, "1d_arith_ret"].values,
            df.loc[df["ticker"] == ticker, "date"],
            args.lbw,
            args.stride,
        )
        for ticker in tickers
    )
    elapsed = time.time() - t0

    out = pd.concat(results, ignore_index=True)
    # Create the 'cpd' subfolder if it does not exist
    cpd_dir = processed_dir / "cpd"
    cpd_dir.mkdir(exist_ok=True)
    out_path = cpd_dir / f"cpd_features_lbw{args.lbw}_s{args.stride}.csv"
    out.to_csv(out_path, index=False)


    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Coverage: {out[f'cpd_nu_{args.lbw}'].notna().mean():.1%} of cells")


if __name__ == "__main__":
    main()