"""CLI entry point: train the Deep Momentum Network on one fold.

Replicates the logic of notebooks/03_dmn_training.ipynb, but headless and parameterized for use by the walk-forward orchestrator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    rets  = pivots[target_cols[0]].values
    vol   = pivots[target_cols[1]].values
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
    use_cpd: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train one fold. Returns checkpoint dict and out-of-sample predictions.
    
    This is the function imported by 03bis_walk_forward.py; the CLI wrapper
    around it is below in main().
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fold_type = cfg.get("fold_type", "expanding")
    fold = cfg[f"folds_{fold_type}"][fold_idx]

    dmn_cfg = cfg["dmn"]
    cpd_lbw     = dmn_cfg["cpd_lbw"]
    seq_len     = dmn_cfg["seq_len"]
    hidden      = dmn_cfg["hidden"]
    dropout     = dmn_cfg["dropout"]
    batch       = dmn_cfg["batch"]
    lr          = dmn_cfg["lr"]
    epochs      = dmn_cfg["epochs"]
    patience    = dmn_cfg["patience"]

    target_vol  = cfg["vol_target"]
    
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    
    feature_cols = [
        "1d_norm_ret", "21d_norm_ret", "63d_norm_ret", "126d_norm_ret", "252d_norm_ret",
        "macd_8_24", "macd_16_48", "macd_32_96",
    ]
    target_cols = ["1d_arith_ret", "60d_ewm_vol"]
    keep_cols = ["date", "ticker"] + target_cols + feature_cols
    
    stocks = pd.read_csv(processed_dir / "stoxx600_processed.csv",
                         parse_dates=["date"], usecols=keep_cols)
    
    
    processed_cpd = PROJECT_ROOT / cfg["data"]["processed_cpd"]
    
    if use_cpd:

        cpd_path = processed_cpd /  f"cpd_features_lbw{cpd_lbw}_s5.csv"
        
        if not cpd_path.exists():
            raise FileNotFoundError(f"Error: CSV file does not exist in {cpd_path}")
        
        cpd = pd.read_csv(cpd_path, parse_dates=["date"])
        stocks = stocks.merge(cpd, on=["date", "ticker"], how="left")
        cpd_cols = [f"cpd_nu_{cpd_lbw}", f"cpd_gamma_{cpd_lbw}"]
        stocks[cpd_cols] = stocks.groupby("ticker")[cpd_cols].ffill()
        feature_cols = feature_cols + cpd_cols
    
    feats, rets, vol, dates, tickers = make_panel(stocks, feature_cols, target_cols)
    
    if verbose:
        print(f"Fold {fold_idx}: train {fold['train_start']}{fold['train_end']}, "
              f"test {fold['test_start']}{fold['test_end']}")
        print(f"Panel: {feats.shape[0]} dates x {feats.shape[1]} tickers x {feats.shape[2]} features")
    
    # Splits
    train_full = np.where(date_mask(dates, fold["train_start"], fold["train_end"]))[0]
    test_idx   = np.where(date_mask(dates, fold["test_start"],  fold["test_end"]))[0]
    n_split = int(0.9 * len(train_full))
    train_idx = train_full[:n_split]
    val_idx   = train_full[n_split:]
    
    X_train, R_train, V_train = build_sequences(feats, rets, vol, train_idx, seq_len, seq_len)
    X_val,   R_val,   V_val   = build_sequences(feats, rets, vol, val_idx,   seq_len, seq_len)
    X_test,  R_test,  V_test  = build_sequences(feats, rets, vol, test_idx,  seq_len, 1)
    
    if verbose:
        print(f"Train sequences: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    
    # Train
    train_loader = DataLoader(TensorDataset(X_train, R_train, V_train),
                              batch_size=batch, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   R_val,   V_val),
                              batch_size=batch, shuffle=False)
    
    model = DeepMomentumNetwork(n_features=len(feature_cols),
                                 hidden_size=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        losses = []
        with torch.set_grad_enabled(train):
            for x, r, v in loader:
                x, r, v = x.to(device), r.to(device), v.to(device)
                positions = model(x)
                loss = sharpe_loss(positions, r, target_vol=target_vol, ex_ante_vol=v)
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
    
    # Predict on test
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
    
    sharpe = strat_ret.mean() / max(strat_ret.std(), 1e-12) * np.sqrt(252)
    
    # Reconstruct (date, ticker) for predictions
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
        "date":   [dates[t] for (t, _) in pairs],
        "ticker": [tickers[a] for (_, a) in pairs],
        "position":   pos_pred,
        "ret":        ret_real,
        "ex_ante_vol": vol_pred,
        "strat_ret":   strat_ret,
    })
    
    # Save
    ckpt_dir = PROJECT_ROOT / cfg["data"]["processed_mod"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"cpd{cpd_lbw}" if use_cpd else "nocpd"
    
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {**cfg, "fold_idx": fold_idx, "use_cpd": use_cpd,
                   "feature_cols": feature_cols},
        "history": history,
        "test_metrics": {"sharpe": float(sharpe),
                         "ann_ret": float(strat_ret.mean() * 252),
                         "ann_vol": float(strat_ret.std() * np.sqrt(252))},
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
    parser.add_argument("--use-cpd", action="store_true",
                        help="Include CPD features as LSTM inputs")
    parser.add_argument("--no-cpd", dest="use_cpd", action="store_false")
    parser.set_defaults(use_cpd=True)
    args = parser.parse_args()
    
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    
    train_dmn(fold_idx=args.fold, cfg=cfg, use_cpd=args.use_cpd)


if __name__ == "__main__":
    main()