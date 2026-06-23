"""Adapter from the trained diffusion model to the hedging pipeline.

Wraps a trained (optionally LoRA fine-tuned) SequentialGaussianDiffusion
checkpoint behind the same two-function interface as volgan_adapter.py
(sample_scenarios + scenarios_to_solver_arrays). The conditioning_history is the
last seq_len-1 (= 21) trading days of observed [log_iv, call_mid_over_s,
log_return] surfaces, shape [seq_len-1, 3, 9, 11], in original (unnormalized)
units; the caller maintains the rolling buffer.

The length is exactly seq_len-1 because SequentialGaussianDiffusion.sample()
validates conditioning of shape [batch, seq_len, *state_shape] and uses the mask
to mark the fixed prefix: we zero-pad to the full seq_len, condition on the
first seq_len-1 positions, and generate the last, matching the training target
convention (conditioning_length=21, target_index=21). Bilinear interpolation and
Black-Scholes pricing are reused from volgan_adapter (grid-agnostic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config.config as config
from diffusion_factor_model import ConditionalTransformer, SequentialGaussianDiffusion
from diffusion_factor_model.fine_tuning import inject_lora
from volgan_adapter import _bilinear_interp, _bs_price


# Default grids from shared_grid_preprocessing.py (11 moneyness × 9 tau)
MONEYNESS_GRID = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])
TAU_GRID = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1.0])

# Channel indices within the [C=3, H=9, W=11] per-timestep state tensor
_CH_LOG_IV = 0
_CH_CALL_OVER_S = 1
_CH_LOG_RETURN = 2


def _infer_shape(data_path: str | Path) -> tuple[int, tuple]:
    data = np.load(data_path)
    if data.ndim != 5:
        raise ValueError(f"Expected 5-D training data [N,S,C,H,W], got shape {data.shape}")
    return int(data.shape[1]), tuple(data.shape[2:])


def _compute_normalization_stats(data_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-element z-score stats (mean, std) over the training set."""
    data = np.load(data_path).astype(np.float32)
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std  = np.where(std == 0, 1.0, std)   # guard zero-variance cells
    return mean, std


def _build_model(
    seq_len: int,
    state_shape: tuple,
    sampling_timesteps: int | None = None,
) -> SequentialGaussianDiffusion:
    model = ConditionalTransformer(
        seq_len=seq_len,
        dim=config.TRANSFORMER_DIM,
        depth=config.TRANSFORMER_LAYERS,
        heads=config.TRANSFORMER_HEADS,
        ff_mult=config.TRANSFORMER_FF_MULT,
        dropout=config.TRANSFORMER_DROPOUT,
        use_bos_token=config.USE_BOS_TOKEN,
        use_alibi=config.USE_ALIBI,
        alibi_slope=config.ALIBI_SLOPE,
        first_token_bias=config.FIRST_TOKEN_BIAS,
        state_shape=state_shape,
    )
    # sampling_timesteps < TIMESTEPS switches the sampler from DDPM to DDIM.
    # DDIM-50 is ~4x faster than the trained DDPM-200 with negligible quality loss
    # for this checkpoint; None keeps the config default (faithful DDPM-200).
    st = config.SAMPLING_TIMESTEPS if sampling_timesteps is None else int(sampling_timesteps)
    return SequentialGaussianDiffusion(
        model,
        seq_len=seq_len,
        timesteps=config.TIMESTEPS,
        sampling_timesteps=st,
        ddim_eta=config.DDIM_ETA,
        objective=config.OBJECTIVE,
        beta_schedule=config.BETA_SCHEDULE,
        auto_normalize=config.AUTO_NORMALIZE,
        state_shape=state_shape,
    )


class DiffusionHedgingAdapter:
    """One-step-ahead scenario generator backed by the team's diffusion model.

    data_path supplies the training window shape and the z-score normalization
    stats. lora_rank is read from the checkpoint hparams when present, so it only
    needs setting to inject LoRA into a base checkpoint. sampling_timesteps=None
    keeps the config default (DDPM-200); set e.g. 50 for the faster DDIM sampler.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_path: str | Path,
        n_scenarios: int = 1000,
        device: str = "cpu",
        lora_rank: int | None = None,
        lora_alpha: float = 16.0,
        sampling_timesteps: int | None = None,
    ) -> None:
        self.n_scenarios = n_scenarios
        self.device = torch.device(device)
        self.sampling_timesteps = sampling_timesteps

        seq_len, state_shape = _infer_shape(data_path)
        self._seq_len = seq_len
        self._state_shape = state_shape

        mean, std = _compute_normalization_stats(data_path)
        self._mean = mean   # [S, C, H, W]
        self._std  = std    # [S, C, H, W]
        # Stats for the generated timestep (last position in the training window)
        self._mean_last = mean[-1]   # [C, H, W]
        self._std_last  = std[-1]    # [C, H, W]

        self._diffusion = self._load(checkpoint_path, seq_len, state_shape, lora_rank, lora_alpha)

    def _load(
        self,
        checkpoint_path: str | Path,
        seq_len: int,
        state_shape: tuple,
        lora_rank: int | None,
        lora_alpha: float,
    ) -> SequentialGaussianDiffusion:
        diffusion = _build_model(seq_len, state_shape, self.sampling_timesteps).to(self.device)

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device)

        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Prefer the LoRA rank stored in the checkpoint over the caller's argument.
        if isinstance(ckpt, dict) and isinstance(ckpt.get("hparams"), dict):
            stored_rank = ckpt["hparams"].get("lora_rank")
            if stored_rank is not None:
                lora_rank = int(stored_rank)

        if lora_rank is not None:
            inject_lora(
                diffusion.model,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=("encoder", "output", "value_proj", "time_mlp"),
            )

        missing, unexpected = diffusion.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading {checkpoint_path}: {unexpected[:10]}")
        if missing:
            # Expected for base checkpoints loaded with LoRA: LoRA weights are new params
            non_lora_missing = [k for k in missing if "lora_" not in k]
            if non_lora_missing:
                raise RuntimeError(f"Non-LoRA keys missing from checkpoint: {non_lora_missing[:10]}")

        diffusion.eval()
        return diffusion


def sample_scenarios(
    adapter: DiffusionHedgingAdapter,
    conditioning_history: np.ndarray,
    spot_current: float,
    N: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N one-step-ahead scenarios from the diffusion model.

    conditioning_history is [seq_len-1, 3, 9, 11] of observed surfaces in
    original (unnormalized) units. Returns (spots_next [N], iv_next [N, 11, 9]),
    where iv_next is absolute IV, moneyness-major.
    """
    n = N if N is not None else adapter.n_scenarios
    history = np.asarray(conditioning_history, dtype=np.float32)
    T = history.shape[0]
    seq_len = adapter._seq_len

    # The sampler requires conditioning with the full seq_len shape and uses the
    # mask to fix the observed prefix. We condition on positions 0..T-1 and
    # generate position T, so the history must be exactly seq_len-1 days long
    # (T = conditioning_length = 21; target_index = 21).
    if T != seq_len - 1:
        raise ValueError(
            f"conditioning_history length {T} must equal seq_len-1 = {seq_len - 1}"
        )
    if history.shape[1:] != adapter._state_shape:
        raise ValueError(
            f"conditioning shape {history.shape[1:]} != model state_shape {adapter._state_shape}"
        )

    # Normalize using per-timestep stats for positions 0..T-1.
    norm = (history - adapter._mean[:T]) / adapter._std[:T]   # [T, 3, 9, 11]
    norm_t = torch.tensor(norm, dtype=torch.float32, device=adapter.device)

    # Zero-pad to full seq_len; the mask, not the padded values, drives behaviour.
    cond_t = torch.zeros(n, seq_len, *adapter._state_shape,
                         dtype=torch.float32, device=adapter.device)
    cond_t[:, :T] = norm_t.unsqueeze(0)                       # broadcast over N
    mask = torch.zeros(n, seq_len, dtype=torch.bool, device=adapter.device)
    mask[:, :T] = True

    with torch.no_grad():
        samples = adapter._diffusion.sample(
            batch_size=n,
            conditioning=cond_t,
            conditioning_mask=mask,
            start_idx=T,
            end_idx=T + 1,
        )
    # Keep only the newly generated position T, then unnormalize.
    new_step = samples[:, T].cpu().numpy()                     # [N, 3, 9, 11]
    new_step = new_step * adapter._std_last + adapter._mean_last

    # Log-return channel is constant across grid cells; recover the next spot.
    log_ret_next = new_step[:, _CH_LOG_RETURN, 0, 0]           # [N]
    spots_next = float(spot_current) * np.exp(log_ret_next)

    # Log-IV channel to absolute IV, transposed from (tau, moneyness) to
    # moneyness-major [N, 11, 9].
    iv_next = np.exp(new_step[:, _CH_LOG_IV]).transpose(0, 2, 1)   # [N, 11, 9]

    return spots_next, iv_next


def scenarios_to_solver_arrays(
    spots_next: np.ndarray,
    iv_next: np.ndarray,
    spot_current: float,
    target_contracts,
    hedge_contracts,
    current_target_values: np.ndarray,
    current_hedge_values: np.ndarray,
    r: float = 0.0,
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
    include_underlying: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Price target and hedge contracts under each scenario into solver arrays.

    Same interface as volgan_adapter.scenarios_to_solver_arrays, with the default
    grids updated to the 11x9 shared grid. Returns (target_changes [N],
    hedge_changes [N, n_h]) as scenario P&L of the portfolio and each instrument.
    When include_underlying is set, the underlying is prepended as instrument 0.
    """
    _CP = "cp_flag"
    _STRIKE = "strike"
    _TAU = "tau"

    all_contracts = list(target_contracts.iterrows()) + list(hedge_contracts.iterrows())
    cp_flags = [str(c[_CP]).upper() for _, c in all_contracts]
    strikes  = np.array([float(c[_STRIKE]) for _, c in all_contracts])
    taus     = np.array([float(c[_TAU]) for _, c in all_contracts])

    m_query = strikes[None, :] / spots_next[:, None]           # [N, n_contracts]

    sigmas      = _bilinear_interp(iv_next, m_grid, tau_grid, m_query, taus)
    prices_next = _bs_price(spots_next, strikes, taus, sigmas, cp_flags, r)

    n_t = len(target_contracts)
    target_changes = prices_next[:, :n_t].sum(axis=1) - float(current_target_values.sum())
    hedge_changes  = prices_next[:, n_t:] - current_hedge_values[None, :]

    if include_underlying:
        # Instrument 0 is the underlying, scenario P&L S_{t+1}^k - S_t. It carries
        # delta=1, vega=0, so the LASSO can hedge delta without buying vega, as in
        # Cont & Vuletić's set where the underlying is always selected (§4.3).
        # Without it the delta hedge is forced through vega-bearing options.
        underlying_change = (spots_next - spot_current)[:, None]
        hedge_changes = np.concatenate([underlying_change, hedge_changes], axis=1)

    return target_changes, hedge_changes
