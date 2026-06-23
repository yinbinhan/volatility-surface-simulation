"""Per-date market-state lookup for the diffusion hedging backtest.

The diffusion model conditions on a rolling 21-day history of 3-channel
[log_iv, call_mid_over_s, log_return] surfaces on the shared 11x9 grid and
generates the next trading day's surface. This module serves that history (and
the date-t surface used to price current instrument values) from the processed
shared-grid tensor the training data was built from. It is the diffusion
analogue of backtest_volgan.build_state_lookup / get_day_state.

Channel/axis conventions mirror prepare_shared_grid_data.build_windows: the
model state is [C=3, H=9 (tau), W=11 (moneyness)], so each per-date [11, 9]
surface is transposed to [9, 11] and the scalar log-return is broadcast across
the grid for channel 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Channel order must match the training data (shared_grid_22d_metadata.json):
#   0 = log_iv, 1 = call_mid_over_s, 2 = log_return_broadcast
_DEFAULT_TENSOR = Path("data/processed_shared_grid_11x9/surface_tensor.npz")

# conditioning_length from the checkpoint's training config (target_index = 21)
CONDITIONING_LENGTH = 21


class DiffusionState:
    """Date-keyed access to shared-grid surfaces for diffusion conditioning."""

    def __init__(self, npz_path: str | Path = _DEFAULT_TENSOR) -> None:
        z = np.load(npz_path, allow_pickle=True)
        self.dates = [pd.Timestamp(str(d)) for d in z["dates"].astype(str)]
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.log_iv = z["log_iv"].astype(np.float32)              # [T, 11, 9]
        self.iv = z["iv"].astype(np.float32)                      # [T, 11, 9]
        self.call_over_s = z["call_mid_over_s"].astype(np.float32)  # [T, 11, 9]
        self.spx_close = z["spx_close"].astype(float)            # [T]
        self.log_return = z["log_return"].astype(np.float32)     # [T]
        self.moneyness_grid = z["moneyness_grid"].astype(float)  # [11]
        self.tau_grid = z["tau_grid"].astype(float)              # [9]

    def get_conditioning_history(
        self, date_t, n_cond: int = CONDITIONING_LENGTH
    ) -> tuple[np.ndarray, float] | None:
        """Return (history [n_cond, 3, 9, 11], spot_t) ending at date_t, or None.

        Conditioning on the n_cond trading days ending at date_t (inclusive)
        generates the next trading day's surface, matching the training
        convention (conditioning_length=21, target = next day). Surfaces are in
        original (unnormalized) units; the adapter normalizes internally.
        """
        idx = self.date_to_idx.get(pd.Timestamp(date_t))
        if idx is None or idx < n_cond - 1:
            return None
        sl = slice(idx - (n_cond - 1), idx + 1)  # [n_cond, 11, 9]
        log_iv = np.transpose(self.log_iv[sl], (0, 2, 1))          # [n_cond, 9, 11]
        call = np.transpose(self.call_over_s[sl], (0, 2, 1))       # [n_cond, 9, 11]
        logret = self.log_return[sl]                               # [n_cond]
        logret_grid = np.broadcast_to(
            logret[:, None, None], (n_cond, 9, 11)
        ).astype(np.float32)
        history = np.stack([log_iv, call, logret_grid], axis=1)    # [n_cond, 3, 9, 11]
        return history.astype(np.float32), float(self.spx_close[idx])

    def get_iv_surface(self, date_t) -> np.ndarray | None:
        """Return the date-t IV surface as [11 (moneyness), 9 (tau)], or None.

        Used to price current instrument values V_t / H_t for the scenario
        increments, consistent with the model's 11x9 scenario surfaces.
        """
        idx = self.date_to_idx.get(pd.Timestamp(date_t))
        if idx is None:
            return None
        return self.iv[idx]  # [11, 9]

    def get_spot(self, date_t) -> float | None:
        idx = self.date_to_idx.get(pd.Timestamp(date_t))
        return float(self.spx_close[idx]) if idx is not None else None

    def has_history(self, date_t, n_cond: int = CONDITIONING_LENGTH) -> bool:
        idx = self.date_to_idx.get(pd.Timestamp(date_t))
        return idx is not None and idx >= n_cond - 1
