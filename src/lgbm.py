"""LightGBM Deep Momentum Network -- V2 (ported from Justine's nb03_dmn_v2.py).

Architecture: 22 features (momentum + region + CPD, each with a lag-1 twin)
-> LightGBM (L2/MSE) -> sigmoid(alpha* x score) -> EMA smoothing -> CPD risk
filter -> positions.

This replaces the previous LightGBM implementation, which trained the same
way as the LSTM (vol-target-scaled positions, z-scored predictions, 3 static
walk-forward folds). V2 instead:

  - trains on 16 annual expanding-window folds (2011-2026), not 3 x 5-year folds
  - uses raw sigmoid(alpha * score) positions -- no vol-target rescaling and
    no cross-sectional z-scoring of predictions
  - uses 22 features (18 momentum incl. lag-1, 2 region, 2 CPD incl. lag-1)
    instead of 8 momentum + 1 sector + 2 CPD (no lags)
  - calibrates alpha on net Sharpe after EMA + transaction costs directly
    (rather than a separate gross/net variant split)

CPD features: Justine's original used stock-level CUSUM + BOCPD severity
scores (nu_cusum_lag1, nu_bocpd_lag1). This project only has the GP-based
changepoint kernel (Wood, Roberts & Zohren 2022) precomputed per lookback
window (cpd_nu_<lbw>, cpd_gamma_<lbw> in data/processed/cpd/), so those two
features are substituted here -- same feature count and role (severity +
location), different underlying method.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


# --- feature set ---

MOMENTUM_HORIZONS = ["1d", "21d", "63d", "126d", "252d"]
MACD_PAIRS = ["8_24", "16_48", "32_96"]


def momentum_feature_cols() -> list[str]:
    cols = [f"{h}_norm_ret" for h in MOMENTUM_HORIZONS]
    cols += [f"{h}_norm_ret_lag1" for h in MOMENTUM_HORIZONS]
    cols += [f"macd_{p}" for p in MACD_PAIRS]
    cols += [f"macd_{p}_lag1" for p in MACD_PAIRS]
    cols += ["ewma_vol", "ewma_vol_lag1"]
    return cols


REGION_FEATURES = ["region_rel_1d", "region_rel_1d_lag1"]


def cpd_feature_cols(cpd_lbw: int) -> list[str]:
    return [f"cpd_nu_{cpd_lbw}_lag1", f"cpd_gamma_{cpd_lbw}_lag1"]


def build_feature_cols(cpd_lbw: int) -> list[str]:
    return momentum_feature_cols() + REGION_FEATURES + cpd_feature_cols(cpd_lbw)


TARGET_COL = "next_return"


# --- data loading ---
#
# The panel and CPD features are cached to parquet after the first build
# (mirrors Justine's parquet-based workflow) since both are expensive to
# rebuild (full-history per-ticker groupby ops / GP changepoint scoring).

def load_panel(cfg: dict, root: Path, force_rebuild: bool = False) -> pd.DataFrame:
    """Build the LightGBM V2 panel from stoxx600_processed.csv (NB01 output).

    Adds: ewma_vol, region_rel_1d, next_return target, and the lag-1 of every
    momentum/region feature. Caches the result to panel_lgbm.parquet.

    Year-boundary fix
    -----------------
    The last trading day of each calendar year has no valid next-day return
    within the same continuous run (the shift(-1) picks up the first trading
    day of the following year, which can be several calendar days later due
    to holidays). We invalidate next_return where the gap exceeds 5 calendar
    days, matching the convention used elsewhere in this project (NB04,
    the LSTM pipeline) so equity curves don't show spurious jumps at fold
    boundaries.
    """
    processed_dir = root / cfg["data"]["processed_dir"]
    cache_path = processed_dir / "panel_lgbm.parquet"
    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    keep_cols = (
        ["date", "ticker", "1d_arith_ret", "60d_ewm_vol", "region"]
        + [f"{h}_norm_ret" for h in MOMENTUM_HORIZONS]
        + [f"macd_{p}" for p in MACD_PAIRS]
    )
    df = pd.read_csv(
        processed_dir / "stoxx600_processed.csv",
        parse_dates=["date"], usecols=keep_cols,
    )
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["ewma_vol"] = df["60d_ewm_vol"]

    df[TARGET_COL] = df.groupby("ticker")["1d_arith_ret"].shift(-1)
    next_date = df.groupby("ticker")["date"].shift(-1)
    gap_days = (next_date - df["date"]).dt.days
    df.loc[gap_days > 5, TARGET_COL] = np.nan
    last_obs = df.groupby("ticker")["date"].transform("max") == df["date"]
    df.loc[last_obs, TARGET_COL] = np.nan

    # Region-relative return (same construction as sector_rel in the
    # existing pipeline, but grouped by region to match Justine's feature).
    region_mean = (df.groupby(["date", "region"])["1d_arith_ret"]
                     .transform("mean")
                     .fillna(0.0))
    df["region_rel_1d"] = df["1d_arith_ret"] - region_mean

    lag_src = (
        [f"{h}_norm_ret" for h in MOMENTUM_HORIZONS]
        + [f"macd_{p}" for p in MACD_PAIRS]
        + ["ewma_vol", "region_rel_1d"]
    )
    for col in lag_src:
        df[f"{col}_lag1"] = df.groupby("ticker")[col].shift(1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def load_cpd_features(cfg: dict, root: Path, cpd_lbw: int, cpd_stride: int,
                      force_rebuild: bool = False) -> pd.DataFrame:
    """Load GP-based CPD features (from scripts/02_compute_cpd.py output) and
    add their lag-1. Caches the result to parquet alongside the source CSV.
    """
    processed_cpd = root / cfg["data"]["processed_cpd"]
    cache_path = processed_cpd / f"cpd_features_lbw{cpd_lbw}_s{cpd_stride}_lgbm.parquet"
    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    csv_path = processed_cpd / f"cpd_features_lbw{cpd_lbw}_s{cpd_stride}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CPD file not found: {csv_path}\n"
            f"Run: python scripts/02_compute_cpd.py --lbw {cpd_lbw} --stride {cpd_stride}"
        )
    nu_col, gamma_col = f"cpd_nu_{cpd_lbw}", f"cpd_gamma_{cpd_lbw}"
    cpd = pd.read_csv(csv_path, parse_dates=["date"])
    cpd = cpd.sort_values(["ticker", "date"])
    cpd[[nu_col, gamma_col]] = cpd.groupby("ticker")[[nu_col, gamma_col]].ffill()
    cpd[f"{nu_col}_lag1"] = cpd.groupby("ticker")[nu_col].shift(1)
    cpd[f"{gamma_col}_lag1"] = cpd.groupby("ticker")[gamma_col].shift(1)

    processed_cpd.mkdir(parents=True, exist_ok=True)
    cpd.to_parquet(cache_path, index=False)
    return cpd


def build_feature_matrix(panel: pd.DataFrame, cpd_features: pd.DataFrame,
                         cpd_lbw: int) -> pd.DataFrame:
    feature_cols = build_feature_cols(cpd_lbw)
    momentum_region_cols = [c for c in feature_cols if c in panel.columns]
    base_cols = ["date", "ticker"] + momentum_region_cols + [TARGET_COL]
    base = panel[base_cols].copy()

    cpd_cols = cpd_feature_cols(cpd_lbw)
    merge_cols = ["date", "ticker"] + [c for c in cpd_cols if c in cpd_features.columns]
    feat = base.merge(cpd_features[merge_cols], on=["date", "ticker"], how="left")

    for col in cpd_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0.0)

    must_have = [f"{MOMENTUM_HORIZONS[0]}_norm_ret", f"{MOMENTUM_HORIZONS[1]}_norm_ret"]
    must_have = [c for c in must_have if c in feat.columns]
    feat = feat.dropna(subset=must_have + [TARGET_COL])
    return feat.sort_values(["ticker", "date"]).reset_index(drop=True)


def input_summary(feat: pd.DataFrame, cpd_lbw: int) -> pd.DataFrame:
    fcols = [c for c in build_feature_cols(cpd_lbw) if c in feat.columns]
    return pd.DataFrame([
        {"item": "rows",       "value": f"{len(feat):,}"},
        {"item": "tickers",    "value": f"{feat['ticker'].nunique():,}"},
        {"item": "date range", "value": f"{feat['date'].min().date()} -> {feat['date'].max().date()}"},
        {"item": "features",   "value": str(len(fcols))},
        {"item": "model",      "value": "LightGBM L2 + calibration alpha* Sharpe-net (V2)"},
        {"item": "target",     "value": TARGET_COL},
    ])


# --- calibration ---
#
# Training: L2/MSE (use_sharpe_loss=False in run_walk_forward). A custom
# Sharpe objective is available below but disabled -- L2 is more stable on
# this weak a signal.
#
# Calibration: alpha* on validation -- position = sigmoid(alpha x score),
# alpha* = argmax net Sharpe(25bps) on validation.

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _sharpe_objective_factory(dates_arr: np.ndarray, eps: float = 1e-8):
    _, inverse, counts = np.unique(dates_arr, return_inverse=True, return_counts=True)
    T = len(counts)
    N_d = counts[inverse].astype(np.float64)

    def _obj(y_pred, dataset):
        y_true = dataset.get_label()
        R = np.bincount(inverse, weights=y_pred * y_true) / counts
        mu = R.mean()
        sigma = R.std() + eps
        coef = np.sqrt(252) * (
            1.0 / (T * sigma)
            - mu * (R - mu) / (max(T - 1, 1) * sigma ** 3)
        )
        grad = -coef[inverse] * y_true / N_d
        hess = np.ones_like(y_pred)
        return grad, hess

    return _obj


def _raw_sharpe(positions: np.ndarray, returns: np.ndarray) -> float:
    p_net = positions * returns
    return float(p_net.mean() / (p_net.std() + 1e-8) * np.sqrt(252))


def _net_sharpe_calibration(pos_flat: np.ndarray, y_flat: np.ndarray,
                            dates: np.ndarray, tickers: np.ndarray,
                            halflife: int, tc: float) -> float:
    ema_a = float(1.0 - np.exp(-np.log(2.0) / halflife))
    df = pd.DataFrame({"date": dates, "ticker": tickers, "pos": pos_flat, "ret": y_flat})
    df = df.sort_values(["ticker", "date"])
    df["pos_s"] = (df.groupby("ticker")["pos"]
                     .transform(lambda s: s.ewm(alpha=ema_a, adjust=False).mean()))
    df["prev"] = df.groupby("ticker")["pos_s"].shift(1)
    df["to"] = (df["pos_s"] - df["prev"]).abs().fillna(0.0)
    df["pnl"] = df["pos_s"] * df["ret"] - tc * df["to"]
    return float(df["pnl"].mean() / (df["pnl"].std() + 1e-8) * np.sqrt(252))


def calibrate_alpha(pred_returns: np.ndarray, y_val: np.ndarray, alpha_max: float,
                    halflife: int, tc: float,
                    dates: np.ndarray | None = None, tickers: np.ndarray | None = None,
                    alphas: np.ndarray | None = None) -> float:
    if alphas is None:
        alphas = np.logspace(0, np.log10(alpha_max), 30)
    use_net = (dates is not None and tickers is not None)
    best_sr, best_a = -np.inf, 1.0
    for a in alphas:
        pos = _sigmoid(a * pred_returns)
        sr = (_net_sharpe_calibration(pos, y_val, dates, tickers, halflife, tc)
              if use_net else _raw_sharpe(pos, y_val))
        if sr > best_sr:
            best_sr = sr
            best_a = float(a)
    return best_a


# --- training ---

def _ic_eval(y_pred, dataset):
    y_true = dataset.get_label()
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    return "IC", (0.0 if np.isnan(corr) else corr), True


def train_fold_lgb(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   lgb_params: dict, n_estimators: int, seed: int,
                   alpha_max: float, position_halflife: int, transaction_cost: float,
                   vl_dates: np.ndarray | None = None, vl_tickers: np.ndarray | None = None,
                   tr_dates: np.ndarray | None = None, use_sharpe_loss: bool = False):
    use_sharpe = use_sharpe_loss and (tr_dates is not None)

    if use_sharpe:
        params = {**lgb_params, "random_state": seed,
                  "objective": _sharpe_objective_factory(tr_dates)}
        feval = _ic_eval
    else:
        params = {**lgb_params, "random_state": seed,
                  "objective": "regression", "metric": "l2"}
        feval = None

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)
    evals_result: dict = {}
    callbacks = [lgb.log_evaluation(period=0), lgb.record_evaluation(evals_result)]

    model = lgb.train(
        params, train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        feval=feval,
        callbacks=callbacks,
    )

    pred_val = model.predict(X_val)
    alpha = calibrate_alpha(
        pred_val, y_val, alpha_max=alpha_max,
        halflife=position_halflife, tc=transaction_cost,
        dates=vl_dates, tickers=vl_tickers,
    )
    return model, alpha, evals_result


# --- walk-forward ---

def walk_forward_splits(feat: pd.DataFrame, test_start: int, window_years: int | None):
    years = sorted(feat["date"].dt.year.unique())
    test_years = [y for y in years if y >= test_start]
    splits = []
    for ty in test_years:
        tr_end = pd.Timestamp(f"{ty - 1}-12-31")
        te_start = pd.Timestamp(f"{ty}-01-01")
        te_end = pd.Timestamp(f"{ty}-12-31")
        tr_start = (pd.Timestamp(f"{ty - 1 - window_years}-01-01")
                    if window_years is not None else None)
        if feat.loc[feat["date"] <= tr_end].shape[0] < 1_000:
            continue
        splits.append({
            "test_year": ty,
            "train_start": tr_start,
            "train_end": tr_end,
            "test_start": te_start,
            "test_end": te_end,
        })
    return splits


def run_walk_forward(feat: pd.DataFrame, feature_cols: list[str], cfg: dict,
                     seed: int = 42, verbose: bool = True):
    lcfg = cfg["lgbm"]
    lgb_params = {
        "learning_rate": lcfg["learning_rate"],
        "max_depth": lcfg["max_depth"],
        "num_leaves": lcfg["num_leaves"],
        "min_child_samples": lcfg["min_child_samples"],
        "subsample": lcfg["subsample"],
        "subsample_freq": 1,
        "colsample_bytree": lcfg["colsample_bytree"],
        "reg_alpha": lcfg["reg_alpha"],
        "reg_lambda": lcfg["reg_lambda"],
        "n_jobs": -1,
        "verbose": -1,
    }
    n_estimators = lcfg["n_estimators"]
    val_frac = lcfg["val_frac"]
    alpha_max = lcfg["alpha_max"]
    position_halflife = lcfg["position_halflife"]
    transaction_cost = lcfg["transaction_cost"]
    test_start = lcfg["test_start"]
    window_years = lcfg["window_years"]

    splits = walk_forward_splits(feat, test_start=test_start, window_years=window_years)
    all_pos = []
    fold_rows = []

    for fold_idx, sp in enumerate(splits):
        ty = sp["test_year"]
        t0 = time.perf_counter()

        mask = feat["date"] <= sp["train_end"]
        if sp.get("train_start") is not None:
            mask &= feat["date"] >= sp["train_start"]
        tr_feat = feat.loc[mask].sort_values(["date", "ticker"])

        X_all = tr_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
        n_val = max(1_000, int(len(tr_feat) * val_frac))
        X_tr, X_vl = X_all[:-n_val], X_all[-n_val:]
        y_all = tr_feat[TARGET_COL].to_numpy(dtype=np.float64)
        y_tr, y_vl = y_all[:-n_val], y_all[-n_val:]

        if len(X_tr) < 500:
            if verbose:
                print(f"  fold {ty} -- too few rows, skipped")
            continue

        tr_slice = tr_feat.iloc[:-n_val]
        vl_slice = tr_feat.iloc[-n_val:]
        model, alpha, _ = train_fold_lgb(
            X_tr, y_tr, X_vl, y_vl,
            lgb_params=lgb_params, n_estimators=n_estimators,
            seed=seed + fold_idx,
            alpha_max=alpha_max, position_halflife=position_halflife,
            transaction_cost=transaction_cost,
            vl_dates=vl_slice["date"].to_numpy(),
            vl_tickers=vl_slice["ticker"].to_numpy(),
            tr_dates=tr_slice["date"].to_numpy(),
            use_sharpe_loss=False,
        )

        pred_vl = model.predict(X_vl)
        p_vl = _sigmoid(alpha * pred_vl)
        val_sharpe = _raw_sharpe(p_vl, y_vl)
        val_ic = float(np.corrcoef(pred_vl, y_vl)[0, 1])

        te_feat = feat.loc[
            (feat["date"] >= sp["test_start"]) & (feat["date"] <= sp["test_end"])
        ].sort_values(["date", "ticker"]).copy()

        if te_feat.empty:
            continue

        X_te = te_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
        pred_te = model.predict(X_te)
        raw_pos = _sigmoid(alpha * pred_te)

        pos_df = te_feat[["date", "ticker"]].copy()
        pos_df["position"] = raw_pos
        all_pos.append(pos_df)

        elapsed = time.perf_counter() - t0
        fold_rows.append({
            "test_year": ty,
            "val_sharpe": round(val_sharpe, 3),
            "val_ic": round(val_ic, 4),
            "alpha_calibrated": round(alpha, 1),
            "n_estimators": model.num_trees(),
            "n_train_rows": len(X_tr),
            "seconds": round(elapsed, 1),
        })

        if verbose:
            print(f"  fold {ty} | val_sharpe={val_sharpe:.3f} | val_ic={val_ic:.4f} | "
                  f"alpha={alpha:.0f} | trees={model.num_trees()} | {elapsed:.0f}s")

    positions_df = (pd.concat(all_pos, ignore_index=True) if all_pos
                    else pd.DataFrame(columns=["date", "ticker", "position"]))
    fold_metrics = pd.DataFrame(fold_rows)
    return positions_df, fold_metrics


# --- SHAP ---

def compute_shap(model: lgb.Booster, feat: pd.DataFrame, feature_cols: list[str],
                 n_sample: int = 5_000, seed: int = 42) -> pd.DataFrame:
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feat), size=min(n_sample, len(feat)), replace=False)
    X_samp = feat.iloc[idx][feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_samp)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return (pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


# --- CPD risk filter ---
#
# When the CPD severity score (nu) is high -> a trend break was detected.
# Positions are pulled back toward 0.5 (neutral) proportionally to nu:
#   pos_filtered = 0.5 + (1 - strength * nu) * (pos - 0.5)

def apply_cpd_filter(positions: pd.DataFrame, feat: pd.DataFrame, cpd_lbw: int,
                     strength: float = 1.0) -> pd.DataFrame:
    nu_col = f"cpd_nu_{cpd_lbw}_lag1"
    if nu_col not in feat.columns:
        return positions
    cpd = feat[["date", "ticker", nu_col]].copy()
    cpd["nu_composite"] = cpd[nu_col].clip(0.0, 1.0)
    merged = positions.merge(cpd[["date", "ticker", "nu_composite"]],
                             on=["date", "ticker"], how="left")
    merged["nu_composite"] = merged["nu_composite"].fillna(0.0)
    confidence = (1.0 - strength * merged["nu_composite"]).clip(0.0, 1.0)
    merged["position"] = 0.5 + confidence * (merged["position"] - 0.5)
    return merged[["date", "ticker", "position"]]


# --- EMA smoothing of positions ---
#
# LightGBM predicts each day independently -> high day-to-day turnover.
# EMA halflife=10d keeps 93% of yesterday's position + 7% of the new one.

def smooth_positions(positions: pd.DataFrame, halflife: int = 10) -> pd.DataFrame:
    alpha = float(1 - np.exp(-np.log(2) / halflife))
    out = positions.sort_values(["ticker", "date"]).copy()
    out["position"] = (
        out.groupby("ticker")["position"]
           .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return out


# --- persistence ---

def save_outputs(cfg: dict, root: Path, positions: pd.DataFrame,
                 fold_metrics: pd.DataFrame) -> pd.DataFrame:
    processed_dir = root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    positions_path = processed_dir / "positions_v2.parquet"
    fold_metrics_path = processed_dir / "fold_metrics_v2.parquet"
    positions.to_parquet(positions_path, index=False)
    fold_metrics.to_parquet(fold_metrics_path, index=False)
    return pd.DataFrame([
        {"output": "positions", "rows": f"{len(positions):,}",
         "path": positions_path.relative_to(root).as_posix()},
        {"output": "fold_metrics", "rows": f"{len(fold_metrics):,}",
         "path": fold_metrics_path.relative_to(root).as_posix()},
    ])
