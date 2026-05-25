"""Run the full out-of-sample backtest by aggregating per-fold predictions.

Headless equivalent of notebooks/04_backtest.ipynb: loads all available
DMN variant predictions, applies transaction costs uniformly, computes
the paper's metrics, and saves tables to data/processed/backtest/.

==============================================================================
USAGE
==============================================================================

Basic invocation (uses YAML defaults for paths, CPD lbw, stride, fold_type):
    python scripts/04_run_backtest.py

Override transaction cost level:
    python scripts/04_run_backtest.py --tc 0.0025

Override fold type:
    python scripts/04_run_backtest.py --fold-type rolling

==============================================================================
WORKED EXAMPLES
==============================================================================

1) Backtest the paper main result (0 bps costs, expanding folds):
       python scripts/04_run_backtest.py --tc 0.0 --fold-type expanding

2) Realistic deployment scenario (25 bps backtest costs, expanding):
       python scripts/04_run_backtest.py --tc 0.0025 --fold-type expanding

3) Rolling-window sensitivity check:
       python scripts/04_run_backtest.py --fold-type rolling --tc 0.0025

4) Generate both expanding and rolling results (run twice):
       python scripts/04_run_backtest.py --fold-type expanding
       python scripts/04_run_backtest.py --fold-type rolling

==============================================================================
OUTPUTS
==============================================================================

CSV tables saved to data/processed/backtest/:
    metrics_raw_{fold_type}.csv          paper Exhibit 3 (raw scale)
    metrics_rescaled_{fold_type}.csv     paper Exhibit 4 (rescaled to 15% vol)
    sharpe_by_year_{fold_type}.csv       year-by-year breakdown

For visualisations and step-by-step analysis, use notebooks/04_backtest.ipynb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers (mirror the notebook functions)
# ---------------------------------------------------------------------------

def add_transaction_costs(df: pd.DataFrame, position_col: str, vol_col: str,
                          gross_col: str, net_col: str,
                          cost: float, target_vol: float) -> pd.DataFrame:
    """Subtract cost * target_vol * |Δ(X/σ)| from the gross strategy return.
    
    Per-ticker turnover; output written to net_col.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["_scaled_pos"]   = df[position_col] / np.maximum(df[vol_col], 1e-6)
    df["_d_scaled_pos"] = df.groupby("ticker")["_scaled_pos"].diff().fillna(0.0)
    df[net_col] = df[gross_col] - cost * target_vol * df["_d_scaled_pos"].abs()
    return df.drop(columns=["_scaled_pos", "_d_scaled_pos"])


def to_portfolio_series(df: pd.DataFrame, strat_col: str) -> pd.Series:
    """Average strategy return across stocks per date."""
    return df.groupby("date")[strat_col].mean().sort_index()


def compute_metrics(returns: pd.Series) -> dict:
    """Standard performance metrics for a daily strategy return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in [
            "Returns", "Vol", "Sharpe", "Downside Dev", "Sortino",
            "MDD", "Calmar", "% +ve", "Avg P / Avg L"
        ]}
    
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    
    downside = r[r < 0]
    dd_dev   = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino  = ann_ret / dd_dev if dd_dev > 0 else np.nan
    
    cum = (1 + r).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
    
    pct_pos = (r > 0).mean()
    avg_p   = r[r > 0].mean()
    avg_l   = abs(r[r < 0].mean())
    p_to_l  = avg_p / avg_l if avg_l > 0 else np.nan
    
    return {
        "Returns": ann_ret, "Vol": ann_vol, "Sharpe": sharpe,
        "Downside Dev": dd_dev, "Sortino": sortino,
        "MDD": mdd, "Calmar": calmar,
        "% +ve": pct_pos, "Avg P / Avg L": p_to_l,
    }


def rescale_to_target_vol(returns: pd.Series, target_vol: float) -> pd.Series:
    """Scale daily returns so the realised annualised vol equals target_vol."""
    r = returns.dropna()
    if len(r) < 2:
        return r
    realised_vol = r.std() * np.sqrt(252)
    if realised_vol == 0:
        return r
    return r * (target_vol / realised_vol)


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty-print formatting for the metrics table."""
    fmt = df.copy()
    for col in ["Returns", "Vol", "Downside Dev", "MDD"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    for col in ["Sharpe", "Sortino", "Calmar", "Avg P / Avg L"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    fmt["% +ve"] = fmt["% +ve"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    return fmt


def yearly_sharpe(returns: pd.Series) -> pd.Series:
    """Compute annualised Sharpe per calendar year."""
    r = returns.dropna()
    return r.groupby(r.index.year).apply(
        lambda y: (y.mean() / y.std()) * np.sqrt(252) if y.std() > 0 else 0.0
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the out-of-sample DMN backtest."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tc", dest="transaction_cost", type=float, default=None,
                        help="Transaction cost (decimal). Default: 0.0025 (25 bps).")
    parser.add_argument("--fold-type", choices=["expanding", "rolling"], default=None,
                        help="Override fold_type set in YAML.")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    
    if args.fold_type:
        cfg["fold_type"] = args.fold_type
    
    # Configuration
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    models_dir    = PROJECT_ROOT / cfg["data"]["processed_mod"]
    target_vol    = cfg["vol_target"]
    cpd_lbw       = cfg["dmn"]["cpd_lbw"]
    cpd_stride    = cfg["dmn"].get("cpd_stride", 1)
    fold_type     = cfg.get("fold_type", "expanding")
    tc = args.transaction_cost if args.transaction_cost is not None else 0.0025

    print(f"Backtest setup: fold_type={fold_type}, cpd_lbw={cpd_lbw}, "
          f"cpd_stride={cpd_stride}, cost={tc*1e4:.0f}bps")

    # Load DMN variants
    variants = {
        "dmn_baseline":         f"{fold_type}_nocpd",
        "dmn_cpd":              f"{fold_type}_cpd{cpd_lbw}_s{cpd_stride}",
        "dmn_cpd_longonly":     f"{fold_type}_cpd{cpd_lbw}_s{cpd_stride}_longonly",
        "dmn_cpd_longonly_tc":  f"{fold_type}_cpd{cpd_lbw}_s{cpd_stride}_longonly_tc25bps",
    }
    
    dmn_predictions = {}
    for label, suffix in variants.items():
        files = sorted(models_dir.glob(f"predictions_fold*_{suffix}.csv"))
        if not files:
            print(f"  {label}: no files matching predictions_fold*_{suffix}.csv — skipping")
            continue
        df = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files],
                        ignore_index=True).sort_values(["date", "ticker"])
        if "strat_ret_gross" not in df.columns:
            df["strat_ret_gross"] = df["strat_ret"].copy()
        dmn_predictions[label] = add_transaction_costs(
            df, position_col="position", vol_col="ex_ante_vol",
            gross_col="strat_ret_gross", net_col="strat_ret",
            cost=tc, target_vol=target_vol,
        )
        print(f"  {label}: {len(df):,} rows, "
              f"{df['date'].min().date()}{df['date'].max().date()}")

    if not dmn_predictions:
        raise FileNotFoundError("No DMN predictions found.")
    
    # Load benchmarks and panel for classical strategies
    benchmarks = pd.read_csv(processed_dir / "benchmark_stoxx600_ew.csv",
                              parse_dates=["date"])
    stocks = pd.read_csv(
        processed_dir / "stoxx600_processed.csv", parse_dates=["date"],
        usecols=["date", "ticker", "1d_arith_ret", "60d_ewm_vol",
                  "252d_arith_ret", "macd_8_24", "macd_16_48", "macd_32_96"],
    )
    oos_start = min(df["date"].min() for df in dmn_predictions.values())
    oos_end   = max(df["date"].max() for df in dmn_predictions.values())
    stocks_oos = stocks.loc[(stocks["date"] >= oos_start) &
                             (stocks["date"] <= oos_end)].copy()
    
    # Classical strategy positions
    stocks_oos["pos_long_only"] = 1.0
    stocks_oos["pos_moskowitz"] = np.sign(stocks_oos["252d_arith_ret"].fillna(0.0))
    macd_signal = stocks_oos[["macd_8_24", "macd_16_48", "macd_32_96"]].mean(axis=1)
    stocks_oos["pos_macd"] = np.sign(macd_signal.fillna(0.0))
    
    for name in ["long_only", "moskowitz", "macd"]:
        stocks_oos[f"strat_{name}_gross"] = (
            stocks_oos[f"pos_{name}"]
            * (target_vol / np.maximum(stocks_oos["60d_ewm_vol"], 1e-6))
            * stocks_oos["1d_arith_ret"].shift(-1)
        )
    
    classical_cols = ["date", "ticker", "60d_ewm_vol"] + \
                     [f"pos_{n}" for n in ["long_only", "moskowitz", "macd"]] + \
                     [f"strat_{n}_gross" for n in ["long_only", "moskowitz", "macd"]]
    classical = (stocks_oos[classical_cols]
                  .dropna(subset=[f"strat_{n}_gross" for n in ["long_only", "moskowitz", "macd"]])
                  .reset_index(drop=True))
    
    for name in ["long_only", "moskowitz", "macd"]:
        classical = add_transaction_costs(
            classical, position_col=f"pos_{name}", vol_col="60d_ewm_vol",
            gross_col=f"strat_{name}_gross", net_col=f"strat_{name}",
            cost=tc, target_vol=target_vol,
        )
    
    # Aggregate to portfolio level
    portfolios = {}
    for label, df in dmn_predictions.items():
        portfolios[f"{label}_gross"] = to_portfolio_series(df, "strat_ret_gross")
        portfolios[f"{label}_net"]   = to_portfolio_series(df, "strat_ret")
    for name in ["long_only", "moskowitz", "macd"]:
        portfolios[f"{name}_gross"] = to_portfolio_series(classical, f"strat_{name}_gross")
        portfolios[f"{name}_net"]   = to_portfolio_series(classical, f"strat_{name}")
    
    sxxr_prices = (benchmarks.loc[benchmarks["benchmark"] == "SXXR"]
                              .set_index("date")["price"].sort_index())
    portfolios["sxxr"] = (sxxr_prices.pct_change()
                                      .loc[(sxxr_prices.index >= oos_start) &
                                           (sxxr_prices.index <= oos_end)])
    
    # Compute and print metrics
    rows = [{**compute_metrics(s), "Strategy": label} for label, s in portfolios.items()]
    metrics_df = (pd.DataFrame(rows).set_index("Strategy")
                  [["Returns", "Vol", "Sharpe", "Downside Dev",
                    "Sortino", "MDD", "Calmar", "% +ve", "Avg P / Avg L"]])
    
    rescaled = {label: rescale_to_target_vol(s, target_vol) for label, s in portfolios.items()}
    rows_r = [{**compute_metrics(s), "Strategy": label} for label, s in rescaled.items()]
    rescaled_metrics_df = (pd.DataFrame(rows_r).set_index("Strategy")
                            [["Returns", "Vol", "Sharpe", "Downside Dev",
                              "Sortino", "MDD", "Calmar", "% +ve", "Avg P / Avg L"]])
    
    yearly_table = pd.DataFrame({
        label: yearly_sharpe(s) for label, s in portfolios.items()
    })

    # Print
    print(f"\n{'='*70}")
    print(f"Performance metrics ({tc*1e4:.0f} bps backtest-time costs)")
    print(f"{'='*70}")
    print(format_metrics(metrics_df).to_string())
    
    print(f"\n{'='*70}")
    print(f"Vol-rescaled to {target_vol:.0%}")
    print(f"{'='*70}")
    print(format_metrics(rescaled_metrics_df).to_string())
    
    # Save with fold_type suffix
    backtest_dir = PROJECT_ROOT / "data" / "processed" / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{fold_type}"
    format_metrics(metrics_df).to_csv(backtest_dir / f"metrics_raw{suffix}.csv")
    format_metrics(rescaled_metrics_df).to_csv(backtest_dir / f"metrics_rescaled{suffix}.csv")
    yearly_table.round(2).to_csv(backtest_dir / f"sharpe_by_year{suffix}.csv")
    
    print(f"\nTables saved to: {backtest_dir}")
    print(f"  metrics_raw{suffix}.csv")
    print(f"  metrics_rescaled{suffix}.csv")
    print(f"  sharpe_by_year{suffix}.csv")


if __name__ == "__main__":
    main()