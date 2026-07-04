"""Train the LightGBM Deep Momentum Network on a single walk-forward fold.

LightGBM is always long-only by construction: sigmoid() maps to (0,1).
It is directly comparable to the LSTM --long-only variants.

Three variants, all produced by this script:

    lgbm_nocpd          Baseline: momentum + sector features only,
                        alpha calibrated on gross Sharpe.

    lgbm_cpd{lbw}_s{s}  Adds GP-CPD severity and location as features,
                        alpha calibrated on gross Sharpe.

    lgbm_cpd{lbw}_s{s}_tc25bps
                        Same features as above, alpha calibrated on
                        NET Sharpe after EMA smoothing and 25bps costs.
                        This is the cost-aware variant, the LightGBM
                        analog of the LSTM's quality-weighted Sharpe loss.

Usage
-----
    python scripts/03_train_lgbm.py --fold 0 --no-cpd
    python scripts/03_train_lgbm.py --fold 0 --use-cpd
    python scripts/03_train_lgbm.py --fold 0 --use-cpd --tc 0.0025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lgbm import (build_feature_cols, TARGET_COL, train_fold_lgb, smooth_positions, apply_cpd_filter, sigmoid)

# Helpers
def date_mask(dates: pd.Series, start: str, end: str) -> pd.Series:
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def build_panel(
    cfg: dict,
    use_cpd: bool,
    cpd_lbw: int,
    cpd_stride: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load processed panel, merge CPD features, compute next-day target.

    Year-boundary fix
    -----------------
    The last trading day of each calendar year has no valid next-day return:
    the shift(-1) picks up the first trading day of the following year,
    which may be 3-5 calendar days later due to holidays. Bloomberg also
    sometimes records spurious multi-day cumulative returns at year boundaries.

    We invalidate any next_return where the gap to the next observation
    exceeds 5 calendar days, matching the convention used in NB04 for the
    LSTM pipeline and the classical benchmarks.

    This fix is applied BEFORE the CPD merge and BEFORE any dropna, so it
    applies identically for all three variants (with or without CPD).
    Without this fix, the corrupted year-end return survives into
    strat_ret_gross and produces visible equity-curve jumps at fold
    boundaries (confirmed by diagnostic: mean_ret on 31-Dec was 0.57,
    compared to ~0.001 on normal days).
    """
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    processed_cpd = PROJECT_ROOT / cfg["data"]["processed_cpd"]

    keep_cols = [
        "date", "ticker", "1d_arith_ret", "60d_ewm_vol",
        "1d_norm_ret", "21d_norm_ret", "63d_norm_ret",
        "126d_norm_ret", "252d_norm_ret",
        "macd_8_24", "macd_16_48", "macd_32_96",
        "1d_arith_ret_rel",
    ]
    df = pd.read_csv(
        processed_dir / "stoxx600_processed.csv",
        parse_dates=["date"],
        usecols=keep_cols,
    )
    df = df.rename(columns={"1d_arith_ret_rel": "sector_rel_ret"})
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Next-day return target
    df[TARGET_COL] = df.groupby("ticker")["1d_arith_ret"].shift(-1)

    # Year-boundary fix (MUST be before CPD merge and before dropna)
    next_date = df.groupby("ticker")["date"].shift(-1)
    gap_days = (next_date - df["date"]).dt.days
    df.loc[gap_days > 5, TARGET_COL] = np.nan
    last_obs = df.groupby("ticker")["date"].transform("max") == df["date"]
    df.loc[last_obs, TARGET_COL] = np.nan

    # CPD merge (after the fix)
    if use_cpd:
        cpd_path = (processed_cpd
                    / f"cpd_features_lbw{cpd_lbw}_s{cpd_stride}.csv")
        if not cpd_path.exists():
            raise FileNotFoundError(
                f"CPD file not found: {cpd_path}\n"
                f"Run: python scripts/02_compute_cpd.py "
                f"--lbw {cpd_lbw} --stride {cpd_stride}"
            )
        if verbose:
            print(f"Using CPD file: {cpd_path.name}")
        cpd = pd.read_csv(cpd_path, parse_dates=["date"])
        df = df.merge(cpd, on=["date", "ticker"], how="left")
        cpd_cols = [f"cpd_nu_{cpd_lbw}", f"cpd_gamma_{cpd_lbw}"]
        df[cpd_cols] = df.groupby("ticker")[cpd_cols].ffill()

    return df


def train_lgbm(
    fold_idx: int,
    cfg: dict,
    use_cpd: Optional[bool] = None,
    transaction_cost: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Train one fold of the LightGBM DMN.

    The three variants differ as follows:

    lgbm_baseline (use_cpd=False, tc=0):
        Features: momentum + sector only.
        Alpha calibration: maximise gross Sharpe on validation.
        This is the fastest variant and the feature-ablation baseline.

    lgbm_cpd (use_cpd=True, tc=0):
        Features: momentum + sector + CPD severity/location.
        Alpha calibration: maximise gross Sharpe on validation.
        Tests whether CPD features add value over pure momentum.

    lgbm_cpd_tc (use_cpd=True, tc=0.0025):
        Features: same as lgbm_cpd.
        Alpha calibration: maximise NET Sharpe after EMA + 25bps costs.
        High alpha → polarised positions → high turnover → penalised.
        This is the cost-aware variant, analogous to the LSTM's
        quality-weighted transaction cost term in the Sharpe loss.
    """
    lgbm_cfg = cfg["lgbm"]
    dmn_cfg = cfg["dmn"]

    use_cpd = use_cpd if use_cpd is not None else True
    transaction_cost = (transaction_cost if transaction_cost is not None
                        else lgbm_cfg.get("transaction_cost", 0.0))

    cpd_lbw = dmn_cfg["cpd_lbw"]
    cpd_stride = dmn_cfg.get("cpd_stride", 1)
    seed = dmn_cfg.get("seed", 42)
    target_vol = cfg["vol_target"]

    n_estimators = lgbm_cfg["n_estimators"]
    val_frac = lgbm_cfg.get("val_frac", 0.10)
    alpha_max = lgbm_cfg.get("alpha_max", 200.0)
    position_halflife = lgbm_cfg.get("position_halflife", 10)
    cpd_filter_strength = lgbm_cfg.get("cpd_filter_strength", 1.0)

    lgb_params = {
        "learning_rate":     lgbm_cfg["learning_rate"],
        "max_depth":         lgbm_cfg["max_depth"],
        "num_leaves":        lgbm_cfg["num_leaves"],
        "min_child_samples": lgbm_cfg["min_child_samples"],
        "subsample":         lgbm_cfg["subsample"],
        "subsample_freq":    1,
        "colsample_bytree":  lgbm_cfg["colsample_bytree"],
        "reg_alpha":         lgbm_cfg["reg_alpha"],
        "reg_lambda":        lgbm_cfg["reg_lambda"],
        "n_jobs":            -1,
        "verbose":           -1,
    }

    np.random.seed(seed)

    fold_type = cfg.get("fold_type", "expanding")
    fold = cfg[f"folds_{fold_type}"][fold_idx]
    feature_cols = build_feature_cols(cpd_lbw, use_cpd=use_cpd)

    panel = build_panel(
        cfg, use_cpd=use_cpd,
        cpd_lbw=cpd_lbw, cpd_stride=cpd_stride,
        verbose=verbose,
    )
    # Drop rows with missing target or missing core momentum features.
    # TARGET_COL is NaN on year-boundary dates (from the fix above),
    # so those rows are automatically excluded here.
    panel = panel.dropna(subset=[TARGET_COL] + feature_cols[:3])

    if verbose:
        print(f"Fold {fold_idx} ({fold_type}): "
              f"train {fold['train_start']}/{fold['train_end']}, "
              f"test  {fold['test_start']}/{fold['test_end']}")
        print(f"Panel: {len(panel):,} rows, {len(feature_cols)} features")
        print(f"use_cpd={use_cpd}, "
              f"tc={transaction_cost*1e4:.0f}bps, seed={seed}")

    # Train/val split: chronological within the training window
    train_mask = date_mask(panel["date"],
                           fold["train_start"], fold["train_end"])
    train_panel = panel.loc[train_mask].sort_values(["date", "ticker"])

    n_val = max(1000, int(len(train_panel) * val_frac))
    tr_slice = train_panel.iloc[:-n_val]
    vl_slice = train_panel.iloc[-n_val:]

    X_train = tr_slice[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_train = tr_slice[TARGET_COL].to_numpy(dtype=np.float64)
    X_val   = vl_slice[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y_val   = vl_slice[TARGET_COL].to_numpy(dtype=np.float64)

    if verbose:
        print(f"Train rows: {len(X_train):,}  Val rows: {len(X_val):,}")

    model, alpha, best_sharpe, evals_result = train_fold_lgb(
        X_train, y_train, X_val, y_val,
        lgb_params=lgb_params,
        n_estimators=n_estimators,
        seed=seed,
        feature_cols=feature_cols,
        vl_dates=vl_slice["date"].to_numpy(),
        vl_tickers=vl_slice["ticker"].to_numpy(),
        alpha_max=alpha_max,
        position_halflife=position_halflife,
        transaction_cost=transaction_cost,   # 0 for baseline/cpd, 0.0025 for tc
    )

    val_ic = float(np.corrcoef(model.predict(X_val), y_val)[0, 1])

    if verbose:
        cost_label = "net" if transaction_cost > 0 else "gross"
        print(f"alpha*={alpha:.1f}  "
              f"val_{cost_label}_sharpe={best_sharpe:.3f}  "
              f"val_IC={val_ic:.4f}  "
              f"trees={model.num_trees()}")

    # Out-of-sample predictions
    test_mask = date_mask(panel["date"],
                          fold["test_start"], fold["test_end"])
    test_panel = panel.loc[test_mask].sort_values(["date", "ticker"]).copy()

    if test_panel.empty:
        raise ValueError(
            f"No test rows for fold {fold_idx} "
            f"({fold['test_start']} – {fold['test_end']})"
        )

    X_test = test_panel[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    pred_test = model.predict(X_test)

    # Z-score predictions cross-sectionally within each date before sigmoid.
    # This must mirror exactly what calibrate_alpha does internally,
    # so that the calibrated alpha is applied to identically-scaled inputs.
    test_panel = test_panel.copy()
    test_panel["_pred"] = pred_test
    test_panel["_pred_z"] = test_panel.groupby("date")["_pred"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8))
    pred_z = test_panel["_pred_z"].to_numpy()
    test_panel = test_panel.drop(columns=["_pred", "_pred_z"])

    raw_pos = sigmoid(alpha * pred_z)

    positions_df = test_panel[["date", "ticker"]].copy()
    positions_df["position"] = raw_pos

    # EMA smoothing
    positions_smooth = smooth_positions(positions_df, halflife=position_halflife)

    # CPD risk filter
    if use_cpd:
        positions_final = apply_cpd_filter(
            positions_smooth, panel,
            cpd_lbw=cpd_lbw, strength=cpd_filter_strength,
        )
    else:
        positions_final = positions_smooth

    # Build result DataFrame — same schema as LSTM prediction files:
    # date, ticker, position, ret, ex_ante_vol, strat_ret_gross, strat_ret
    result_df = positions_final.merge(
        test_panel[["date", "ticker", TARGET_COL, "60d_ewm_vol"]],
        on=["date", "ticker"], how="left",
    ).rename(columns={TARGET_COL: "ret", "60d_ewm_vol": "ex_ante_vol"})

    # Year-boundary fix on test panel (belt-and-suspenders: the NaN was
    # already set in build_panel but the merge may reintroduce edge cases)
    result_df = result_df.sort_values(["ticker", "date"])
    next_date_test = result_df.groupby("ticker")["date"].shift(-1)
    gap_test = (next_date_test - result_df["date"]).dt.days
    result_df.loc[gap_test > 5, "ret"] = np.nan
    last_test = (
        result_df.groupby("ticker")["date"].transform("max") == result_df["date"]
    )
    result_df.loc[last_test, "ret"] = np.nan

    # Vol-scaled gross strategy return (paper Eq. 11)
    target_vol_scale = target_vol / np.maximum(result_df["ex_ante_vol"], 1e-6)
    result_df["strat_ret_gross"] = (
        result_df["position"] * target_vol_scale * result_df["ret"]
    )

    # Net return after per-ticker turnover costs
    result_df = result_df.sort_values(["ticker", "date"])
    scaled_pos = result_df["position"] / np.maximum(result_df["ex_ante_vol"], 1e-6)
    turnover = scaled_pos.groupby(result_df["ticker"]).diff().abs().fillna(0.0)
    result_df["strat_ret"] = (
        result_df["strat_ret_gross"]
        - transaction_cost * target_vol * turnover
    )
    result_df = result_df.sort_values(["date", "ticker"]).reset_index(drop=True)


    # Metrics
    gross = result_df["strat_ret_gross"].dropna()
    net   = result_df["strat_ret"].dropna()
    gross_sharpe = (gross.mean() / max(gross.std(), 1e-12)) * np.sqrt(252)
    net_sharpe   = (net.mean()   / max(net.std(),   1e-12)) * np.sqrt(252)

    if verbose:
        print(f"OOS Sharpe — gross: {gross_sharpe:+.3f}, "
              f"net: {net_sharpe:+.3f}")

    
    # Persist model and predictions
    suffix_parts = [fold_type]
    suffix_parts.append(
        f"lgbm_cpd{cpd_lbw}_s{cpd_stride}" if use_cpd else "lgbm_nocpd"
    )
    if transaction_cost > 0:
        suffix_parts.append(f"tc{int(transaction_cost * 1e4)}bps")
    suffix = "_".join(suffix_parts)

    ckpt_dir = PROJECT_ROOT / cfg["data"]["processed_mod"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ckpt_dir / f"lgbm_fold{fold_idx}_{suffix}.txt"))
    result_df.to_csv(
        ckpt_dir / f"predictions_fold{fold_idx}_{suffix}.csv", index=False
    )

    return {
        "model":          model,
        "alpha":          alpha,
        "fold_idx":       fold_idx,
        "use_cpd":        use_cpd,
        "transaction_cost": transaction_cost,
        "val_ic":         val_ic,
        "test_metrics": {
            "sharpe_gross": float(gross_sharpe),
            "sharpe_net":   float(net_sharpe),
            "val_ic":       val_ic,
        },
    }


# CLI
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--use-cpd",  dest="use_cpd", action="store_true",
                        default=None)
    parser.add_argument("--no-cpd",   dest="use_cpd", action="store_false")
    parser.add_argument("--tc", dest="transaction_cost", type=float,
                        default=None,
                        help="Transaction cost, e.g. 0.0025 for 25 bps")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    train_lgbm(
        fold_idx=args.fold,
        cfg=cfg,
        use_cpd=args.use_cpd,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    main()