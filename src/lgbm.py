"""LightGBM-based Deep Momentum Network (alternative to the LSTM in src/dmn.py).

Pipeline:
    features → LightGBM (L2/MSE) → sigmoid(alpha* x score) → EMA smoothing
    → CPD risk filter → positions

This mirrors the role of src/dmn.py for the LSTM variant. The key design
choices that make it comparable to the LSTM:

1. POSITION SIZING: sigmoid(alpha* x score) where alpha* is calibrated on
   the validation set to maximise net-of-cost Sharpe. This is the LightGBM
   analog of the LSTM's end-to-end Sharpe training — instead of
   backpropagating through the cost term, we search for the scaling factor
   that maximises the same objective on held-out data.

2. COST-AWARE CALIBRATION: the tc variant calibrates alpha against
   net Sharpe after EMA smoothing and 25bps costs, penalising alpha values
   that produce high turnover. This is the LightGBM analog of the LSTM's
   quality-weighted transaction cost term.

3. EMA SMOOTHING: halflife=10d reduces day-to-day position changes,
   analogous to the LSTM's implicit temporal smoothing via its 63-day
   sequence window. This is the primary turnover control mechanism.

4. NO RANK-NORMALISATION: positions are sigmoid(alpha* x raw_score),
   not rank-normalised. Rank-normalisation would destroy the alpha
   calibration and make all variants identical regardless of features
   or cost awareness. The jump problem at fold boundaries is handled
   instead by (a) invalidating year-boundary returns in build_panel()
   and (b) zeroing fold-boundary turnover in add_transaction_costs() in nb04.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# Feature set
MOMENTUM_FEATURES = [
    "1d_norm_ret", "21d_norm_ret", "63d_norm_ret",
    "126d_norm_ret", "252d_norm_ret",
    "macd_8_24", "macd_16_48", "macd_32_96",
]

SECTOR_FEATURES = ["sector_rel_ret"]


def cpd_feature_cols(cpd_lbw: int) -> list[str]:
    return [f"cpd_nu_{cpd_lbw}", f"cpd_gamma_{cpd_lbw}"]


def build_feature_cols(cpd_lbw: int, use_cpd: bool = True) -> list[str]:
    cols = list(MOMENTUM_FEATURES) + list(SECTOR_FEATURES)
    if use_cpd:
        cols += cpd_feature_cols(cpd_lbw)
    return cols

TARGET_COL = "next_return"


# Sigmoid helper
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


# Sharpe metrics used for alpha calibration
def _raw_sharpe(positions: np.ndarray, returns: np.ndarray) -> float:
    """Gross annualised Sharpe of position * return."""
    pnl = positions * returns
    std = pnl.std()
    if std < 1e-12:
        return 0.0
    return float(pnl.mean() / std * np.sqrt(252))


def _net_sharpe_with_ema_and_costs(
    raw_pos: np.ndarray,
    ret: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    halflife: int,
    transaction_cost: float,
) -> float:
    """Net annualised Sharpe after EMA smoothing and per-ticker turnover costs.

    This is the objective that the cost-aware alpha calibration maximises.
    It mirrors the LSTM's quality-weighted Sharpe loss: the penalty for
    turnover is embedded in the objective used to select the model's
    effective position-sizing parameter (alpha).

    Parameters
    ----------
    raw_pos : array of sigmoid(alpha * score) before smoothing
    ret     : next-day return for each (date, ticker) observation
    dates, tickers : arrays for groupby operations
    halflife : EMA halflife in trading days
    transaction_cost : one-way cost as a decimal (e.g. 0.0025 for 25 bps)
    """
    ema_alpha = float(1.0 - np.exp(-np.log(2.0) / halflife))
    df = pd.DataFrame({
        "date": dates, "ticker": tickers,
        "pos": raw_pos, "ret": ret,
    })
    df = df.sort_values(["ticker", "date"])
    df["pos_smooth"] = (
        df.groupby("ticker")["pos"]
          .transform(lambda s: s.ewm(alpha=ema_alpha, adjust=False).mean())
    )
    df["prev_pos"] = df.groupby("ticker")["pos_smooth"].shift(1)
    df["turnover"] = (df["pos_smooth"] - df["prev_pos"]).abs().fillna(0.0)
    df["pnl"] = df["pos_smooth"] * df["ret"] - transaction_cost * df["turnover"]
    std = df["pnl"].std()
    if std < 1e-12:
        return 0.0
    return float(df["pnl"].mean() / std * np.sqrt(252))


# Alpha calibration
def calibrate_alpha(
    pred_scores: np.ndarray,
    y_val: np.ndarray,
    alpha_max: float = 10.0,
    dates: np.ndarray | None = None,
    tickers: np.ndarray | None = None,
    halflife: int = 10,
    transaction_cost: float = 0.0,
) -> tuple[float, float]:
    """Find alpha* = argmax Sharpe(sigmoid(alpha * z_score(score))).

    LightGBM predicts daily returns, which are tiny (order 1e-4 to 1e-3).
    sigmoid(alpha * 0.0005) ≈ 0.5 for any alpha in [1, 200] — positions
    collapse to ~0.5 for every stock, all variants become indistinguishable.

    Z-scoring cross-sectionally within each date maps predictions to a
    [-3, +3] range where sigmoid produces meaningful dispersion (~0.05 to
    ~0.95). Alpha then controls aggressiveness on a sensible scale:
        alpha=0.5 → positions in (0.18, 0.82)  conservative
        alpha=1.0 → positions in (0.05, 0.95)  moderate
        alpha=2.0 → positions in (0.007, 0.993) aggressive

    The cost-aware variant (tc>0) is penalised for high alpha because
    polarised positions → high turnover → lower net Sharpe.
    """
    # Z-score cross-sectionally within each date
    if dates is not None:
        df_tmp = pd.DataFrame({
            "date": dates,
            "score": pred_scores,
            "ret": y_val,
        })
        df_tmp["score_z"] = df_tmp.groupby("date")["score"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        scores = df_tmp["score_z"].to_numpy()
        rets = df_tmp["ret"].to_numpy()
    else:
        scores = (pred_scores - pred_scores.mean()) / (pred_scores.std() + 1e-8)
        rets = y_val

    # Alpha grid: 0.1 to 10 (z-scores are in [-3,3], so alpha>5 already
    # gives near-binary positions — no need to search up to 200)
    alphas = np.logspace(-1,  np.log10(alpha_max), 40)

    use_net = (
        transaction_cost > 0.0
        and dates is not None
        and tickers is not None
    )

    best_sharpe = -np.inf
    best_alpha = 1.0

    for a in alphas:
        pos = sigmoid(a * scores)
        if use_net:
            sr = _net_sharpe_with_ema_and_costs(
                pos, rets, dates, tickers, halflife, transaction_cost
            )
        else:
            sr = _raw_sharpe(pos, rets)

        if sr > best_sharpe:
            best_sharpe = sr
            best_alpha = float(a)

    return best_alpha, best_sharpe


# LightGBM training
def _ic_eval(y_pred: np.ndarray, dataset: lgb.Dataset):
    """Information coefficient — more stable than RMSE on noisy returns."""
    y_true = dataset.get_label()
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    return "IC", (0.0 if np.isnan(corr) else corr), True


def train_fold_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lgb_params: dict,
    n_estimators: int,
    seed: int,
    feature_cols: list[str] | None = None,
    vl_dates: np.ndarray | None = None,
    vl_tickers: np.ndarray | None = None,
    alpha_max: float = 200.0,
    position_halflife: int = 10,
    transaction_cost: float = 0.0,
) -> tuple[lgb.Booster, float, float, dict]:
    """Train one LightGBM fold and calibrate the position-sizing alpha.

    Returns
    -------
    model        : trained LightGBM booster
    alpha        : calibrated alpha*
    best_sharpe  : validation Sharpe at alpha* (net if tc>0, gross otherwise)
    evals_result : training/validation loss curves
    """
    params = {
        **lgb_params,
        "random_state": seed,
        "objective": "regression",
        "metric": "l2",
    }

    train_set = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_cols or "auto",
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val,
        reference=train_set,
        feature_name=feature_cols or "auto",
        free_raw_data=False,
    )

    evals_result: dict = {}
    callbacks = [
        lgb.log_evaluation(period=0),
        lgb.record_evaluation(evals_result),
    ]

    model = lgb.train(
        params, train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        feval=_ic_eval,
        callbacks=callbacks,
    )

    pred_val = model.predict(X_val)
    alpha, best_sharpe = calibrate_alpha(
        pred_val, y_val,
        alpha_max=alpha_max,
        dates=vl_dates,
        tickers=vl_tickers,
        halflife=position_halflife,
        transaction_cost=transaction_cost,
    )

    return model, alpha, best_sharpe, evals_result



# Position post-processing
def smooth_positions(positions: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """EMA-smooth raw daily positions per ticker to reduce turnover.

    LightGBM predicts each day independently, producing noisy day-to-day
    position changes. EMA smoothing (halflife in trading days) reduces
    turnover while preserving the directional signal, analogous to the
    LSTM's implicit temporal smoothing via its recurrent hidden state.
    """
    ema_alpha = float(1.0 - np.exp(-np.log(2.0) / halflife))
    out = positions.sort_values(["ticker", "date"]).copy()
    out["position"] = (
        out.groupby("ticker")["position"]
           .transform(lambda s: s.ewm(alpha=ema_alpha, adjust=False).mean())
    )
    return out


def apply_cpd_filter(
    positions: pd.DataFrame,
    feat: pd.DataFrame,
    cpd_lbw: int,
    strength: float = 1.0,
) -> pd.DataFrame:
    """Pull positions toward neutral (0.5) when CPD severity is high.

        pos_filtered = 0.5 + (1 - strength * nu) * (pos - 0.5)

    nu in [0,1] is the CPD severity (cpd_nu_{lbw}). strength=0 disables
    the filter; strength=1 fully neutralises the position when nu=1.
    This acts as a risk filter: when a regime break is detected, the
    model reduces its directional bet regardless of the raw signal.
    """
    nu_col = f"cpd_nu_{cpd_lbw}"
    if nu_col not in feat.columns:
        return positions

    cpd = feat[["date", "ticker", nu_col]].copy()
    cpd[nu_col] = cpd[nu_col].clip(0.0, 1.0)
    merged = positions.merge(cpd, on=["date", "ticker"], how="left")
    merged[nu_col] = merged[nu_col].fillna(0.0)
    confidence = (1.0 - strength * merged[nu_col]).clip(0.0, 1.0)
    merged["position"] = 0.5 + confidence * (merged["position"] - 0.5)
    return merged[["date", "ticker", "position"]]



# SHAP
def compute_shap(
    model: lgb.Booster,
    feat: pd.DataFrame,
    feature_cols: list[str],
    n_sample: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature on a random sample."""
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feat), size=min(n_sample, len(feat)), replace=False)
    X_sample = (feat.iloc[idx][feature_cols]
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )