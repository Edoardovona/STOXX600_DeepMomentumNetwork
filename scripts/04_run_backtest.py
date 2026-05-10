"""Run the full out-of-sample backtest by aggregating per-fold predictions
and producing the paper's performance tables.

Reads predictions_fold*.csv files written by 03_train_dmn.py and computes:
- Annualised return, vol, Sharpe, Sortino, Calmar
- Maximum drawdown
- Hit rate, profit-to-loss ratio
- Equity curve and drawdown plots
- Comparison vs benchmarks (long-only, optionally MACD)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_metrics(strat_ret: pd.Series) -> dict:
    """Standard performance metrics on a daily strategy return series."""
    daily = strat_ret.dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    
    downside = daily[daily < 0]
    downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = ann_ret / downside_dev if downside_dev > 0 else np.nan
    
    cum = (1 + daily).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    mdd = drawdown.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    
    pos_returns = (daily > 0).sum()
    pct_positive = pos_returns / len(daily)
    avg_p = daily[daily > 0].mean()
    avg_l = abs(daily[daily < 0].mean())
    p_to_l = avg_p / avg_l if avg_l > 0 else np.nan
    
    return {
        "Returns":  ann_ret, "Vol": ann_vol, "Sharpe": sharpe,
        "Downside Dev": downside_dev, "Sortino": sortino,
        "MDD": mdd, "Calmar": calmar,
        "% +ve": pct_positive, "Ave P / Ave L": p_to_l,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", default="cpd21",
                        help="Suffix of predictions files (e.g. 'cpd21', 'nocpd')")
    args = parser.parse_args()
    
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    models_dir    = processed_dir / "models"
    
    # ---- Load all per-fold predictions ----
    pred_files = sorted(models_dir.glob(f"predictions_fold*_{args.variant}.parquet"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found for variant '{args.variant}'.")
    
    print(f"Found {len(pred_files)} fold(s):")
    all_preds = []
    for f in pred_files:
        df = pd.read_parquet(f)
        print(f"  {f.name}: {len(df):,} rows, {df['date'].min().date()}{df['date'].max().date()}")
        all_preds.append(df)
    preds = pd.concat(all_preds, ignore_index=True).sort_values(["date", "ticker"])
    
    # Aggregate to portfolio level (equal-weighted across stocks per day)
    portfolio_ret = preds.groupby("date")["strat_ret"].mean().sort_index()
    
    # Compute metrics 
    metrics = compute_metrics(portfolio_ret)
    
    print(f"\nFull out-of-sample backtest ({args.variant}, "
          f"{portfolio_ret.index[0].date()}{portfolio_ret.index[-1].date()}):")
    for k, v in metrics.items():
        if "+ve" in k or "Ratio" in k or "P / L" in k:
            print(f"  {k:20s}: {v:.3f}")
        else:
            print(f"  {k:20s}: {v:+.2%}" if "Vol" in k or "Returns" in k or "Dev" in k or "MDD" in k
                  else f"  {k:20s}: {v:+.3f}")
    
    # Plot equity curve
    cum = (1 + portfolio_ret).cumprod()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(cum.index, cum.values, lw=1.2, label=f"DMN ({args.variant})")
    axes[0].axhline(1.0, color="gray", lw=0.5)
    axes[0].set_title(f"Out-of-sample equity curve ({args.variant})  "
                       f"Sharpe={metrics['Sharpe']:+.2f}")
    axes[0].set_ylabel("Cumulative return"); axes[0].legend()
    
    drawdown = (cum - cum.cummax()) / cum.cummax()
    axes[1].fill_between(drawdown.index, drawdown.values, 0,
                          color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown"); axes[1].set_xlabel("")
    plt.tight_layout()
    
    out_path = models_dir / f"backtest_{args.variant}.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nEquity curve saved: {out_path}")


if __name__ == "__main__":
    main()