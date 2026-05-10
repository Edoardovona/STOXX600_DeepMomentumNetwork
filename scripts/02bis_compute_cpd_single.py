"""Pre-compute CPD features (severity nu, location gamma) for a SINGLE ticker.

Faster than 02_compute_cpd.py because it skips the panel loop. Useful for end-to-end testing of the DMN pipeline
before committing compute to the full stock run.

Output format matches scripts/02_compute_cpd.py exactly, so notebook 03 can load it with the same merge logic.

Usage:
    python scripts/02bis_compute_cpd_single.py --ticker "TTE FP" --lbw 21 --stride 5
    
    foreach ($t in "TTE FP", "SAP GY", "AZN LN", "NESN SE") {
    python scripts/02bis_compute_cpd_single.py --ticker "$t" --lbw 21 --stride 5
}
"""
# change here the lbw and stride (i.e. CPD score is recomputed every 5 days as default)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cpd import cpd_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TTE FP",
                        help="Single Bloomberg ticker (e.g. 'TTE FP')")
    parser.add_argument("--lbw", type=int, default=21,
                        help="Lookback window for the GP CPD module")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride between consecutive CPD windows")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    csv_path = processed_dir / "stoxx600_processed.csv"

    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        usecols=["date", "ticker", "1d_arith_ret"],
    )
    one = (df.loc[df["ticker"] == args.ticker]
             .sort_values("date").reset_index(drop=True))

    if one.empty:
        available = sorted(df["ticker"].unique())[:30]
        raise ValueError(
            f"Ticker '{args.ticker}' not found. First 30 available:\n  {available}"
        )

    returns = one["1d_arith_ret"].values
    dates   = one["date"]
    n = len(returns)
    print(f"Ticker {args.ticker}: {n} days ({dates.iloc[0].date()}/{dates.iloc[-1].date()})")

    nu    = np.full(n, np.nan)
    gamma = np.full(n, np.nan)

    valid_ends = list(range(args.lbw, n, args.stride))
    print(f"Computing {len(valid_ends)} windows (lbw={args.lbw}, stride={args.stride})")

    t_start = time.time()
    n_failed = 0
    for i, t in enumerate(valid_ends):
        window = returns[t - args.lbw : t]
        if np.isnan(window).any():
            continue
        try:
            nu_t, gamma_t = cpd_scores(window, args.lbw)
            nu[t] = nu_t
            gamma[t] = gamma_t
        except Exception:
            n_failed += 1

        # Progress every ~10% of total
        if (i + 1) % max(1, len(valid_ends) // 10) == 0:
            pct = 100 * (i + 1) / len(valid_ends)
            elapsed = time.time() - t_start
            print(f"  {pct:3.0f}% done  ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t_start
    n_done = (~np.isnan(nu)).sum()

    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Successful: {n_done} / {len(valid_ends)} windows; "
          f"failures: {n_failed}")
    print(f"  nu     stats: min={np.nanmin(nu):.3f}, "
          f"median={np.nanmedian(nu):.3f}, max={np.nanmax(nu):.3f}")
    print(f"  gamma  stats: min={np.nanmin(gamma):.3f}, "
          f"median={np.nanmedian(gamma):.3f}, max={np.nanmax(gamma):.3f}")

    # Save in the same long format as the full pipeline
    out = pd.DataFrame({
        "date": dates.values,
        "ticker": args.ticker,
        f"cpd_nu_{args.lbw}":    nu,
        f"cpd_gamma_{args.lbw}": gamma,
    })

    safe_ticker = args.ticker.replace(" ", "_").replace("/", "_")
    cpd_dir = processed_dir / "cpd"
    cpd_dir.mkdir(exist_ok=True)
    out_path = cpd_dir / f"cpd_features_lbw{args.lbw}_s{args.stride}_SINGLE_{safe_ticker}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e3:.1f} KB)")
    print(f"\nIn notebook 03, set:")
    print(f"  TEST_TICKER = \"{args.ticker}\"")
    print(f"  CPD_LBW     = {args.lbw}")


if __name__ == "__main__":
    main()