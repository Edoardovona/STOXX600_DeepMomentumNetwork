"""Deep Momentum Network: LSTM-based architecture for time-series momentum,
following Lim, Zohren & Roberts (2019) and Wood, Roberts & Zohren (2022).

The model takes per-asset feature sequences and outputs positions in (-1, 1)
via a tanh head. Training optimizes the (negative) Sharpe ratio of the
resulting strategy returns, end-to-end.

Reference equations:
    Eq. 11  -- Volatility-scaled strategy return
    Eq. 13  -- LSTM forward pass producing position sequences
    Eq. 14  -- Sharpe-ratio loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepMomentumNetwork(nn.Module):
    """Single-layer LSTM + tanh head producing positions in (-1, 1).

    Inputs
    ------
    x : (batch, seq_len, n_features) tensor of stock-level features.

    Outputs
    -------
    positions : (batch, seq_len) tensor of trading positions in (-1, 1).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # nn.LSTM dropout only applies between stacked layers
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, hidden_size)
        h, _ = self.lstm(x)
        h = self.dropout(h)
        # Squash to (-1, 1); squeeze the singleton output dim
        return torch.tanh(self.head(h)).squeeze(-1)


def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    target_vol: float = 0.15,
    ex_ante_vol: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative annualised Sharpe ratio of the volatility-scaled strategy.

    Implements paper Eq. 14, with the volatility scaling of Eq. 11:
        R_{t+1} = X_t * (sigma_tgt / sigma_t) * r_{t+1}

    Parameters
    ----------
    positions : (batch, seq_len) positions in (-1, 1) -- output of the model.
    returns   : (batch, seq_len) realized 1d arithmetic returns r_{t+1}
                aligned so positions[t] is held over returns[t].
    target_vol : annualised target volatility (default 15% as in paper).
    ex_ante_vol : (batch, seq_len) ex-ante annualised volatility estimate
                (e.g. 60d EWMA vol). If None, defaults to 1.0 (no scaling).
    eps : numerical floor for the volatility denominator.

    Returns
    -------
    Negative Sharpe ratio (scalar tensor) suitable for SGD minimisation.
    """
    if ex_ante_vol is None:
        scaled = positions * returns
    else:
        scaled = positions * (target_vol / (ex_ante_vol + eps)) * returns

    # Pool across all (asset, time) pairs in the minibatch
    flat = scaled.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    mean = flat.mean()
    std = flat.std() + eps
    sharpe = mean / std * (252.0 ** 0.5)
    return -sharpe