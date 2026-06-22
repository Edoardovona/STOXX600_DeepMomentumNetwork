"""Deep Momentum Network: LSTM-based architecture for time-series momentum,
following Lim, Zohren & Roberts (2019) and Wood, Roberts & Zohren (2022).

The model takes per-asset feature sequences and outputs positions either in (-1, 1) or (0, 1).
Training optimizes the (negative) Sharpe ratio of the resulting strategy returns, end-to-end.

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
    transaction_cost: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative annualised Sharpe ratio of the volatility-scaled strategy.

    If transaction_cost > 0, subtracts the cost of turnover from the realised
    return at each step (paper Eq. C1). The cost is on |Δ(X/sigma)|, i.e. the
    change in the vol-scaled (unleveraged) position, not on the raw position.

    --------------------------------------------------------------------
    Quality-weighted cost (extension over the paper's Eq. C1):

    The original formula charges a flat cost on every unit of turnover,
    regardless of whether the trade was a good idea. This means a position
    change that correctly anticipates a move pays the same marginal cost as
    a noisy, unproductive flip — there is no incentive to trade when the
    expected edge clearly outweighs the cost.

    We weight the cost by trade "conviction": whether the position change
    moved in the direction the realised return subsequently confirms.
        agreement = sign(delta_position) * sign(realised_return)
            +1  ->  the trade was directionally correct (confirmed)
            -1  ->  the trade was directionally wrong (contradicted)
             0  ->  no clear relationship
    A confirmed trade is charged less (cost discounted towards 0); a
    contradicted trade is charged close to the full flat cost. This lets
    the model place a trade "no matter the transaction cost" when the
    realised outcome justifies it, while still penalising noisy churn.

    This only activates when transaction_cost > 0; with transaction_cost=0
    the loss reduces exactly to the paper's original Sharpe loss.
    --------------------------------------------------------------------
    """
    if ex_ante_vol is None:
        scaled_pos = positions
        scaled_ret = positions * returns
    else:
        scaled_pos = positions / (ex_ante_vol + eps)
        scaled_ret = positions * (target_vol / (ex_ante_vol + eps)) * returns

    if transaction_cost > 0:
        prev_scaled = torch.cat([torch.zeros_like(scaled_pos[:, :1]), scaled_pos[:, :-1]], dim=1)
        delta = scaled_pos - prev_scaled
        turnover = delta.abs()

        # Trade quality weighting (see docstring above)
        agreement = torch.sign(delta) * torch.sign(returns)
        quality_multiplier = torch.sigmoid(-agreement)  # temperature fixed at 1.0

        cost = transaction_cost * target_vol * turnover * quality_multiplier
        scaled_ret = scaled_ret - cost

    flat = scaled_ret.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    mean = flat.mean()
    std = flat.std() + eps
    sharpe = mean / std * (252 ** 0.5)
    return -sharpe