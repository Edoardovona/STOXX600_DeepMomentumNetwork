"""Pre-compute CPD features (severity nu, location gamma) for a SINGLE ticker.

Faster than 02_compute_cpd.py because it skips the panel loop. Useful for end-to-end
testing of the DMN pipeline before committing compute to the full stock run.

Output format matches scripts/02_compute_cpd.py exactly, so notebook 03 can 
load it with the same merge logic.

==============================================================================
USAGE
==============================================================================

All parameters can be set in configs/default.yaml under the `dmn:` section
(`cpd_lbw`, `cpd_stride`), or overridden via the command line. CLI flags take
precedence over YAML values.

Basic invocation (uses YAML defaults, ticker 'TTE FP'):
    python scripts/02bis_compute_cpd_single.py --ticker "TTE FP"

Override LBW and stride:
    python scripts/02bis_compute_cpd_single.py --ticker "TTE FP" --lbw 21 --stride 5

==============================================================================
WORKED EXAMPLES
==============================================================================

1) Test with YAML defaults on a specific ticker:
       python scripts/02bis_compute_cpd_single.py --ticker "SAP GY"

2) Multi-ticker PowerShell loop (overrides LBW and stride):
       foreach ($t in "TTE FP", "SAP GY", "AZN LN", "NESN SE") {
           python scripts/02bis_compute_cpd_single.py --ticker "$t" --lbw 21 --stride 5
       }

==============================================================================
OUTPUTS
==============================================================================

Saved to data/processed/cpd/ with self-documenting filenames:
    cpd_features_lbw<LBW>_s<STRIDE>_<TICKER>.csv

Examples:
    cpd_features_lbw21_s5_TTE_FP.csv
    cpd_features_lbw63_s1_SAP_GY.csv
"""

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
    parser = argparse.ArgumentParser(
        description="Pre-compute CPD features for a SINGLE STOXX 600 ticker."
    )
    parser.add_argument("--ticker", default="TTE FP",
                        help="Single Bloomberg ticker (e.g. 'TTE FP')")
    parser.add_argument("--lbw", type=int, default=None,
                        help="Lookback window for the GP CPD module "
                             "(default: cfg.dmn.cpd_lbw from YAML)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between consecutive CPD windows "
                             "(default: cfg.dmn.cpd_stride from YAML)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI args override YAML; YAML provides the defaults
    dmn_cfg = cfg.get("dmn", {})
    lbw    = args.lbw    if args.lbw    is not None else dmn_cfg.get("cpd_lbw", 21)
    stride = args.stride if args.stride is not None else dmn_cfg.get("cpd_stride", 1)
    # If you don't pass --stride in the terminal, the script goes to dmn_cfg  and looks for cpd_stride (5)

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

    valid_ends = list(range(lbw, n, stride))
    print(f"Computing {len(valid_ends)} windows (lbw={lbw}, stride={stride})")

    t_start = time.time()
    n_failed = 0
    for i, t in enumerate(valid_ends):
        window = returns[t - lbw : t]
        if np.isnan(window).any():
            continue
        try:
            nu_t, gamma_t = cpd_scores(window, lbw)
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
    print(f"  nu    stats: min={np.nanmin(nu):.3f}, "
          f"median={np.nanmedian(nu):.3f}, max={np.nanmax(nu):.3f}")
    print(f"  gamma  stats: min={np.nanmin(gamma):.3f}, "
          f"median={np.nanmedian(gamma):.3f}, max={np.nanmax(gamma):.3f}")

    # Save in the same long format as the full pipeline
    out = pd.DataFrame({
        "date": dates.values,
        "ticker": args.ticker,
        f"cpd_nu_{lbw}":    nu,
        f"cpd_gamma_{lbw}": gamma,
    })

    safe_ticker = args.ticker.replace(" ", "_").replace("/", "_")
    
    # Use the proper YAML path for the cpd directory (same as full panel)
    cpd_dir = PROJECT_ROOT / cfg["data"]["processed_cpd"]
    cpd_dir.mkdir(parents=True, exist_ok=True)
    
    # Matching the new file pattern logic
    out_path = cpd_dir / f"cpd_features_lbw{lbw}_s{stride}_{safe_ticker}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e3:.1f} KB)")
    print(f"\nIn notebook 03, set:")
    print(f"  TEST_TICKER = \"{args.ticker}\"")
    print(f"  CPD_LBW     = {lbw}")


if __name__ == "__main__":
    main()