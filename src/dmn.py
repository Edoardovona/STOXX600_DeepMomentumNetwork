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
    """Single-layer LSTM head producing trading positions.

    The output activation depends on `long_only`:
        - long_only=False (default, paper):     tanh    → positions in (-1, 1)
        - long_only=True (long-only framework): sigmoid → positions in  (0, 1)

    Inputs
    ------
    x : (batch, seq_len, n_features) tensor of stock-level features.

    Outputs
    -------
    positions : (batch, seq_len) tensor of trading positions, in (-1, 1) if
        long_only=False or (0, 1) if long_only=True.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 20,
        dropout: float = 0.3,
        long_only: bool = False,
    ) -> None:
        super().__init__()
        self.long_only = long_only
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
        # Squash to (-1, 1) [paper] or (0, 1) [long-only]; squeeze the singleton output dim
        out = self.head(h).squeeze(-1)
        return torch.sigmoid(out) if self.long_only else torch.tanh(out)

def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    target_vol: float = 0.15,
    ex_ante_vol: torch.Tensor | None = None,
    transaction_cost: float = 0.0,    # cost per unit change in scaled position
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative annualised Sharpe ratio of the volatility-scaled strategy.
    
    If transaction_cost > 0, subtracts the cost of turnover from the realised return at each step (paper Eq. C1).
    The cost is on |Δ(X/sigma)| -- i.e., the change in the vol-scaled (unleveraged) position -- not on the raw position.
    """
    if ex_ante_vol is None:
        scaled_pos = positions
        scaled_ret = positions * returns
    else:
        scaled_pos = positions / (ex_ante_vol + eps)
        scaled_ret = positions * (target_vol / (ex_ante_vol + eps)) * returns
    
    if transaction_cost > 0:
        # Turnover: change in scaled position. First step has no t-1; pad with zeros.
        prev_scaled = torch.cat([torch.zeros_like(scaled_pos[:, :1]),
                                  scaled_pos[:, :-1]], dim=1)
        turnover = (scaled_pos - prev_scaled).abs()
        cost = transaction_cost * target_vol * turnover
        scaled_ret = scaled_ret - cost
    
    flat = scaled_ret.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)
    
    mean = flat.mean()
    std = flat.std() + eps
    sharpe = mean / std * (252.0 ** 0.5)
    return -sharpe