"""Train the Deep Momentum Network (DMN) on a single walk-forward fold.

This is the CLI entry point for training one fold. For training all folds in sequence,
use scripts/03bis_walk_forward.py (which imports and calls train_dmn() from this module).

==============================================================================
USAGE
==============================================================================

All flags can be set in configs/default.yaml under the `dmn:` section, or
overridden via the command line. CLI flags take precedence over YAML values.

Basic invocation:
    python scripts/03_train_dmn.py --fold 0

Fold selection:
    --fold N             Train fold index N (default: 0)
    --config PATH        Use a different YAML config (default: configs/default.yaml)

Model variants:
    --use-cpd            Include CPD severity/location as LSTM inputs (default)
    --no-cpd             Train baseline DMN without the CPD module
    --long-only          Constrain positions to (0, 1) via sigmoid activation
    --no-long-only       Allow long-short positions in (-1, 1) via tanh (paper)
    --tc 0.0025          Apply 25 bps transaction-cost penalty inside the loss
                          (default: 0.0; use 0.0025 for the realistic deployment)

==============================================================================
WORKED EXAMPLES
==============================================================================

1) Paper baseline (no CPD), fold 0:
       python scripts/03_train_dmn.py --fold 0 --no-cpd --no-long-only

2) Paper main result (with CPD), fold 1:
       python scripts/03_train_dmn.py --fold 1 --use-cpd --no-long-only

3) Long-only adaptation with CPD, fold 2:
       python scripts/03_train_dmn.py --fold 2 --use-cpd --long-only

4) Realistic deployment (long-only + 25 bps costs), fold 3:
       python scripts/03_train_dmn.py --fold 3 --use-cpd --long-only --tc 0.0025

5) Rolling-window variant (set fold_type=rolling in YAML beforehand) with CPD:
       python scripts/03_train_dmn.py --fold 0 --use-cpd

==============================================================================
OUTPUTS
==============================================================================

Saved to data/processed/models/ with self-documenting filenames:

    dmn_<fold_type>_fold<i>_<suffix>.pt
    predictions_<fold_type>_fold<i>_<suffix>.csv

where <suffix> encodes the experiment configuration:

    cpd21                          paper main model
    nocpd                          paper baseline (LSTM only)
    cpd21_longonly                 long-only adaptation
    cpd21_longonly_tc25bps         realistic deployment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.dmn import DeepMomentumNetwork, sharpe_loss


def date_mask(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def make_panel(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> tuple:
    df = df.dropna(subset=target_cols)
    pivots = {col: df.pivot(index="date", columns="ticker", values=col)
              for col in feature_cols + target_cols}
    dates   = pivots[target_cols[0]].index
    tickers = pivots[target_cols[0]].columns
    feats = np.stack([pivots[c].values for c in feature_cols], axis=-1)
    # Shift returns by -1 so rets[t] is the return earned from t to t+1
    rets_raw = pivots[target_cols[0]].values
    rets = np.roll(rets_raw, shift=-1, axis=0)
    rets[-1] = np.nan
    vol = pivots[target_cols[1]].values
    return (feats.astype(np.float32), rets.astype(np.float32),
            vol.astype(np.float32), dates, tickers)


def build_sequences(feats, rets, vol, date_idx, seq_len, stride):
    n_dates, n_assets, n_features = feats.shape
    X_list, R_list, V_list = [], [], []
    valid_ends = [t for t in date_idx if t - seq_len + 1 >= 0]
    valid_ends = valid_ends[::stride]
    for t in valid_ends:
        x_block = feats[t - seq_len + 1 : t + 1]
        r_block = rets[t - seq_len + 1 : t + 1]
        v_block = vol[t - seq_len + 1 : t + 1]
        for a in range(n_assets):
            x_a, r_a, v_a = x_block[:, a, :], r_block[:, a], v_block[:, a]
            if np.isfinite(x_a).all() and np.isfinite(r_a).all() and np.isfinite(v_a).all():
                X_list.append(x_a); R_list.append(r_a); V_list.append(v_a)
    if not X_list:
        raise ValueError("No valid sequences extracted.")
    return (torch.from_numpy(np.stack(X_list)),
            torch.from_numpy(np.stack(R_list)),
            torch.from_numpy(np.stack(V_list)))


def train_dmn(
    fold_idx: int,
    cfg: dict,
    use_cpd: Optional[bool] = None,
    long_only: Optional[bool] = None,
    transaction_cost: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Train one fold of the DMN.

    Arguments override the corresponding YAML values when not None.
    Returns the checkpoint dict; persists model + predictions to disk.
    """
    dmn_cfg = cfg["dmn"]

    # Resolve overridable flags: CLI/orchestrator arg → YAML → default
    use_cpd          = use_cpd          if use_cpd          is not None else True
    long_only        = long_only        if long_only        is not None else dmn_cfg.get("long_only", False)
    transaction_cost = transaction_cost if transaction_cost is not None else dmn_cfg.get("transaction_cost", 0.0)

    # Static hyperparameters
    cpd_lbw    = dmn_cfg["cpd_lbw"]
    cpd_stride = dmn_cfg.get("cpd_stride", 1)
    seq_len    = dmn_cfg["seq_len"]
    hidden     = dmn_cfg["hidden"]
    dropout    = dmn_cfg["dropout"]
    batch      = dmn_cfg["batch"]
    lr         = dmn_cfg["lr"]
    epochs     = dmn_cfg["epochs"]
    patience   = dmn_cfg["patience"]
    seed       = dmn_cfg.get("seed", 42)
    target_vol = cfg["vol_target"]

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fold resolution
    fold_type = cfg.get("fold_type", "expanding")
    fold = cfg[f"folds_{fold_type}"][fold_idx]

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    processed_cpd = PROJECT_ROOT / cfg["data"]["processed_cpd"]
    ckpt_dir      = PROJECT_ROOT / cfg["data"]["processed_mod"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load processed panel
    feature_cols = [
        "1d_norm_ret", "21d_norm_ret", "63d_norm_ret", "126d_norm_ret", "252d_norm_ret",
        "macd_8_24", "macd_16_48", "macd_32_96",
    ]
    target_cols = ["1d_arith_ret", "60d_ewm_vol"]
    keep_cols   = ["date", "ticker"] + target_cols + feature_cols

    stocks = pd.read_csv(processed_dir / "stoxx600_processed.csv",
                         parse_dates=["date"], usecols=keep_cols)

    if use_cpd:
        cpd_path = processed_cpd / f"cpd_features_lbw{cpd_lbw}_s{cpd_stride}.csv"
        if not cpd_path.exists():
            raise FileNotFoundError(
                f"CPD feature file not found: {cpd_path}.\n"
                f"Run: python scripts/02_compute_cpd.py --lbw {cpd_lbw} --stride {cpd_stride}"
            )
        if verbose:
            print(f"Using CPD file: {cpd_path.name}")

        cpd = pd.read_csv(cpd_path, parse_dates=["date"])
        stocks = stocks.merge(cpd, on=["date", "ticker"], how="left")
        cpd_cols = [f"cpd_nu_{cpd_lbw}", f"cpd_gamma_{cpd_lbw}"]
        stocks[cpd_cols] = stocks.groupby("ticker")[cpd_cols].ffill()
        feature_cols = feature_cols + cpd_cols

    feats, rets, vol, dates, tickers = make_panel(stocks, feature_cols, target_cols)

    if verbose:
        print(f"Fold {fold_idx} ({fold_type}): "
              f"train {fold['train_start']}/{fold['train_end']}, "
              f"test {fold['test_start']}/{fold['test_end']}")
        print(f"Panel: {feats.shape[0]} dates x {feats.shape[1]} tickers x "
              f"{feats.shape[2]} features")
        print(f"Flags: use_cpd={use_cpd}, long_only={long_only}, "
              f"tc={transaction_cost*1e4:.0f}bps, seed={seed}")

    # Splits
    train_full = np.where(date_mask(dates, fold["train_start"], fold["train_end"]))[0]
    test_idx   = np.where(date_mask(dates, fold["test_start"],  fold["test_end"]))[0]
    n_split   = int(0.9 * len(train_full))
    train_idx = train_full[:n_split]
    val_idx   = train_full[n_split:]

    X_train, R_train, V_train = build_sequences(feats, rets, vol, train_idx, seq_len, seq_len)
    X_val,   R_val,   V_val   = build_sequences(feats, rets, vol, val_idx,   seq_len, seq_len)
    X_test,  R_test,  V_test  = build_sequences(feats, rets, vol, test_idx,  seq_len, 1)

    if verbose:
        print(f"Sequences: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # Model + optimiser
    train_loader = DataLoader(TensorDataset(X_train, R_train, V_train),
                              batch_size=batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   R_val,   V_val),
                              batch_size=batch, shuffle=False)

    model = DeepMomentumNetwork(
        n_features=len(feature_cols),
        hidden_size=hidden,
        dropout=dropout,
        long_only=long_only,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        losses = []
        with torch.set_grad_enabled(train):
            for x, r, v in loader:
                x, r, v = x.to(device), r.to(device), v.to(device)
                positions = model(x)
                loss = sharpe_loss(
                    positions, r,
                    target_vol=target_vol,
                    ex_ante_vol=v,
                    transaction_cost=transaction_cost,
                )
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    history = {"train": [], "val": []}
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        train_loss = run_epoch(train_loader, train=True)
        val_loss   = run_epoch(val_loader,   train=False)
        history["train"].append(train_loss); history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or no_improve >= patience):
            print(f"Epoch {epoch:3d}: train Sharpe={-train_loss:+.3f}, "
                  f"val Sharpe={-val_loss:+.3f}  (best={-best_val:+.3f})")

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Out-of-sample predictions
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, R_test, V_test),
                             batch_size=batch, shuffle=False)
    all_pos, all_ret, all_vol = [], [], []
    with torch.no_grad():
        for x, r, v in test_loader:
            x = x.to(device)
            pos = model(x).cpu().numpy()
            all_pos.append(pos[:, -1])
            all_ret.append(r[:, -1].numpy())
            all_vol.append(v[:, -1].numpy())

    pos_pred = np.concatenate(all_pos)
    ret_real = np.concatenate(all_ret)
    vol_pred = np.concatenate(all_vol)
    strat_ret = pos_pred * (target_vol / np.maximum(vol_pred, 1e-6)) * ret_real
    # reporting only gross results, leaving cost-aware accounting entirely to the backtest layer
    # i.e., the 25 bps only enters the training loss, not the test metric.
    sharpe = strat_ret.mean() / max(strat_ret.std(), 1e-12) * np.sqrt(252)

    # Reconstruct (date, ticker) pairs for the test predictions
    valid_ends = [t for t in test_idx if t - seq_len + 1 >= 0]
    pairs = []
    for t in valid_ends:
        x_block = feats[t - seq_len + 1 : t + 1]
        r_block = rets[t - seq_len + 1 : t + 1]
        v_block = vol[t - seq_len + 1 : t + 1]
        for a in range(feats.shape[1]):
            if (np.isfinite(x_block[:, a, :]).all()
                and np.isfinite(r_block[:, a]).all()
                and np.isfinite(v_block[:, a]).all()):
                pairs.append((t, a))

    result_df = pd.DataFrame({
        "date":         [dates[t] for (t, _) in pairs],
        "ticker":       [tickers[a] for (_, a) in pairs],
        "position":     pos_pred,
        "ret":          ret_real,
        "ex_ante_vol":  vol_pred,
        "strat_ret":    strat_ret,
    })

    # Build descriptive suffix for filenames
    if use_cpd:
        suffix_parts = [fold_type, f"cpd{cpd_lbw}_s{cpd_stride}"]
    else:
        suffix_parts = [fold_type, "nocpd"]
    if long_only:
        suffix_parts.append("longonly")
    if transaction_cost > 0:
        suffix_parts.append(f"tc{int(transaction_cost*1e4)}bps")
    suffix = "_".join(suffix_parts)

    # Persist
    ckpt = {
        "state_dict":   model.state_dict(),
        "config":       {**cfg, "fold_idx": fold_idx, "use_cpd": use_cpd,
                          "long_only": long_only, "transaction_cost": transaction_cost,
                          "feature_cols": feature_cols},
        "history":      history,
        "test_metrics": {"sharpe":  float(sharpe),
                         "ann_ret": float(strat_ret.mean() * 252),
                         "ann_vol": float(strat_ret.std()  * np.sqrt(252))},
    }
    torch.save(ckpt, ckpt_dir / f"dmn_fold{fold_idx}_{suffix}.pt")
    result_df.to_csv(ckpt_dir / f"predictions_fold{fold_idx}_{suffix}.csv", index=False)

    if verbose:
        print(f"Done. Test Sharpe: {sharpe:+.3f}")

    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--use-cpd", dest="use_cpd", action="store_true", default=None)
    parser.add_argument("--no-cpd",  dest="use_cpd", action="store_false")
    parser.add_argument("--long-only",    dest="long_only", action="store_true", default=None)
    parser.add_argument("--no-long-only", dest="long_only", action="store_false")
    parser.add_argument("--tc", dest="transaction_cost", type=float, default=None,
                        help="Transaction cost (decimal, e.g. 0.0025 for 25 bps)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    train_dmn(
        fold_idx=args.fold,
        cfg=cfg,
        use_cpd=args.use_cpd,
        long_only=args.long_only,
        transaction_cost=args.transaction_cost,
    )


if __name__ == "__main__":
    main()