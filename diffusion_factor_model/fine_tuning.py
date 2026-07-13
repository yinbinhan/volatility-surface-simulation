import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint as _grad_ckpt


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    Keeps the base linear layer frozen and learns a low-rank update.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(
            torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


    @property
    def weight(self):
        # Expose a Linear-compatible weight attribute for modules (e.g. MultiheadAttention)
        # that access `.weight` directly on projection layers.
        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.base.weight + delta

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(p in name for p in patterns)


def inject_lora(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] = ("encoder", "output", "value_proj", "time_mlp"),
    prefix: str = "",
) -> nn.Module:
    """Replace selected nn.Linear layers with LoRA-augmented layers."""

    for child_name, child in list(module.named_children()):
        fq_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches_any(fq_name, target_modules):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            continue
        inject_lora(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            prefix=fq_name,
        )
    return module


def freeze_non_lora_params(module: nn.Module):
    for name, param in module.named_parameters():
        param.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-prob, summed over non-batch dims."""

    log2pi = math.log(2 * math.pi)
    out = -0.5 * (((x - mean) ** 2) * torch.exp(-log_var) + log_var + log2pi)
    return out.flatten(start_dim=1).sum(dim=-1)


def gaussian_kl_divergence(
    mean_new: torch.Tensor,
    log_var_new: torch.Tensor,
    mean_ref: torch.Tensor,
    log_var_ref: torch.Tensor,
) -> torch.Tensor:
    """KL(N_new || N_ref), summed over non-batch dims."""

    var_new = torch.exp(log_var_new)
    var_ref = torch.exp(log_var_ref)
    kl = 0.5 * (
        log_var_ref - log_var_new
        + (var_new + (mean_new - mean_ref) ** 2) / var_ref
        - 1.0
    )
    return kl.flatten(start_dim=1).sum(dim=-1)


@dataclass
class FineTuneStats:
    loss: float
    policy_loss: float
    kl_loss: float
    reward_mean: float
    reward_std: float
    grad_norm: float




class ArbitrageValidator:
    """Arbitrage-violation calculator for generated volatility/price surfaces."""

    def __init__(self, grid_m, grid_t, *, strict_grid_match: bool = False, eps: float = 1e-8):
        self.m = torch.tensor(grid_m, dtype=torch.float32)
        self.t = torch.tensor(grid_t, dtype=torch.float32)
        if self.m.ndim != 1 or self.t.ndim != 1:
            raise ValueError("grid_m and grid_t must be 1D arrays")
        if len(self.m) < 2 or len(self.t) < 2:
            raise ValueError("grid_m and grid_t must have at least two points")
        if not torch.all(self.m[1:] > self.m[:-1]):
            raise ValueError("grid_m must be strictly increasing")
        if not torch.all(self.t[1:] > self.t[:-1]):
            raise ValueError("grid_t must be strictly increasing")

        self.dt = self.t[1:] - self.t[:-1]
        self.dm = self.m[1:] - self.m[:-1]
        self.strict_grid_match = strict_grid_match
        self.eps = float(eps)

    def per_sample_total_violation(self, tensor_5d: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor_5d, np.ndarray):
            prices = torch.tensor(tensor_5d, dtype=torch.float32)
        else:
            prices = tensor_5d

        device = prices.device
        t = self.t.to(device)
        dt = self.dt.to(device)
        dm = self.dm.to(device)

        # expects input [B, S, C, H, W], channel-1 = normalized call price C/S
        if prices.ndim != 5:
            raise ValueError(f"Expected [B,S,C,H,W], got shape {tuple(prices.shape)}")
        batch_size, seq_len, channels, height, width = prices.shape
        if channels < 2:
            raise ValueError(
                f"ArbitrageValidator expects at least 2 channels and uses channel index 1 for price, got C={channels}"
            )

        # Grid handling for resolution mismatch between configured grid and generated surfaces.
        if height != len(t) or width != len(self.m):
            if self.strict_grid_match:
                raise ValueError(
                    f"Grid/data resolution mismatch: got HxW={height}x{width}, expected {len(t)}x{len(self.m)}. "
                    "Set strict_grid_match=False to auto-resample grid."
                )
            if height != len(t):
                t = torch.linspace(float(t[0].item()), float(t[-1].item()), steps=height, device=device, dtype=prices.dtype)
                dt = t[1:] - t[:-1]
            if width != len(self.m):
                m = torch.linspace(float(self.m[0].item()), float(self.m[-1].item()), steps=width, device=device, dtype=prices.dtype)
                dm = m[1:] - m[:-1]

        prices = prices[:, :, 1, :, :].reshape(batch_size * seq_len, height, width)

        diff_t = prices[:, :-1, :] - prices[:, 1:, :]
        l1_map = torch.relu(t[:-1].view(1, -1, 1) * diff_t / dt.clamp_min(self.eps).view(1, -1, 1))
        l1 = torch.sum(l1_map, dim=(1, 2))

        diff_m = prices[:, :, 1:] - prices[:, :, :-1]
        slope = diff_m / dm.clamp_min(self.eps).view(1, 1, -1)
        l2_map = torch.relu(slope)
        l2 = torch.sum(l2_map, dim=(1, 2))

        l3_map = torch.relu(slope[:, :, :-1] - slope[:, :, 1:])
        l3 = torch.sum(l3_map, dim=(1, 2))

        total = l1 + l2 + l3
        return total.view(batch_size, seq_len).mean(dim=1)

    def calculate_violations(self, tensor_5d: torch.Tensor):
        penalties = self.per_sample_total_violation(tensor_5d)
        return {
            "total": float(penalties.mean().item()),
            "per_sample": penalties.detach().cpu().numpy(),
        }


def make_arbitrage_reward_fn(validator: ArbitrageValidator):
    """Return reward function for RL fine-tuning: maximize negative total violation."""

    def reward_fn(samples: torch.Tensor) -> torch.Tensor:
        penalties = validator.per_sample_total_violation(samples)
        return -penalties

    return reward_fn


# Grid for the 22-day shared-grid IV surfaces. MUST stay identical to
# gen_arbitrage_ccdf.py (validated to 6 significant figures against
# day22_violation_summary.csv). Layout is [tau (rows), moneyness (cols)].
SHARED_GRID_MONEYNESS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4)
SHARED_GRID_TAU = (0.0027397260273972603, 0.019230769230769232, 0.038461538461538464,
                   0.08333333333333333, 0.16666666666666666, 0.25, 0.5, 0.75, 1.0)


def make_arbitrage_reward_fn_iv(m_grid=SHARED_GRID_MONEYNESS, tau_grid=SHARED_GRID_TAU,
                                R: float = 0.0, eps: float = 1e-8):
    """Arbitrage reward for 2-channel (log-IV, return) surfaces.

    Reconstructs the normalized call-price surface C/S from channel 0 (log-IV)
    via Black-Scholes (R=0 by default) on the real (moneyness, tau) grid, then
    penalizes L1 (calendar), L2 (vertical) and L3 (butterfly) static-arbitrage
    violations. Differentiable torch port of cs_surface + violations in
    gen_arbitrage_ccdf.py. Input surfaces are [B, S, C, H=tau, W=moneyness];
    channel 0 is log-IV. Returns reward = -mean_seq(L1+L2+L3), shape [B].
    """
    m_t = torch.tensor(m_grid, dtype=torch.float32)
    tau_t = torch.tensor(tau_grid, dtype=torch.float32)
    dm_t = (m_t[1:] - m_t[:-1]).clamp_min(eps)
    dt_t = (tau_t[1:] - tau_t[:-1]).clamp_min(eps)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def reward_fn(samples: torch.Tensor) -> torch.Tensor:
        dev = samples.device
        m = m_t.to(dev); tau = tau_t.to(dev); dm = dm_t.to(dev); dt = dt_t.to(dev)
        iv = torch.exp(samples[:, :, 0])                        # [B, S, tau, m]
        TT = tau.view(1, 1, -1, 1)
        MM = m.view(1, 1, 1, -1)
        sq = iv * torch.sqrt(TT)
        d1 = (-torch.log(MM) + (R + 0.5 * iv ** 2) * TT) / sq
        d2 = d1 - sq
        ncdf = lambda x: 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
        price = ncdf(d1) - MM * torch.exp(-R * TT) * ncdf(d2)   # [B, S, tau, m]

        diff_t = price[:, :, :-1, :] - price[:, :, 1:, :]
        l1 = torch.relu(tau[:-1].view(1, 1, -1, 1) * diff_t / dt.view(1, 1, -1, 1)).sum(dim=(-2, -1))
        diff_m = price[:, :, :, 1:] - price[:, :, :, :-1]
        slope = diff_m / dm.view(1, 1, 1, -1)
        l2 = torch.relu(slope).sum(dim=(-2, -1))
        l3 = torch.relu(slope[:, :, :, :-1] - slope[:, :, :, 1:]).sum(dim=(-2, -1))
        total = (l1 + l2 + l3).mean(dim=1)                      # [B]
        return -total

    return reward_fn


class OnlineDDPMLoRAFineTuner:
    """Online policy-gradient fine-tuning for sequential DDPM with LoRA.

    - Generates fresh rollouts each step (no replay buffer).
    - Uses sequence-level rewards supplied by an external callback with direct differentiable optimization.
    - Regularizes policy against a frozen reference with score-space KL surrogate.
    """

    def __init__(
        self,
        diffusion,
        *,
        reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        lr: float = 1e-4,
        kl_weight: float = 1e-3,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Sequence[str] = ("encoder", "output", "value_proj", "time_mlp"),
        normalize_rewards: bool = True,
        max_grad_norm: float = 1.0,
        transition_chunk_size: int = 0,
        grad_checkpoint: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.diffusion = diffusion
        self.device = device or diffusion.device
        self.reward_fn = reward_fn
        self.kl_weight = float(kl_weight)
        self.normalize_rewards = normalize_rewards
        self.max_grad_norm = float(max_grad_norm)
        self.transition_chunk_size = int(transition_chunk_size)
        self.grad_checkpoint = bool(grad_checkpoint)

        inject_lora(
            self.diffusion.model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        freeze_non_lora_params(self.diffusion.model)

        self.reference = copy.deepcopy(self.diffusion).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        trainable_params = [p for p in self.diffusion.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable LoRA parameters found. Check lora_target_modules.")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    def _compute_transition_params(self, model, context, key_padding_mask, target_indices, x_t, times):
        pred_noise, x_start = model._model_predictions(
            context,
            key_padding_mask,
            target_indices,
            x_t,
            times,
        )
        mean, _, _ = model.q_posterior(x_start, x_t, times)
        return mean, pred_noise, x_start

    def rollout(self, batch_size: int, show_progress: bool = False):
        model = self.diffusion
        seq_context = torch.zeros((batch_size, model.seq_len, *model.state_shape), device=self.device)
        generated_states: List[torch.Tensor] = []
        transitions: List[Dict[str, torch.Tensor]] = []

        positions = range(model.seq_len)
        if show_progress:
            positions = tqdm(positions, desc="FT rollout positions", leave=False)

        for pos in positions:
            x_t = torch.randn((batch_size, *model.state_shape), device=self.device)
            target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)

            for t in reversed(range(model.num_timesteps)):
                times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                context, key_padding_mask = model.build_context(seq_context, target_indices, x_t)

                if self.grad_checkpoint:
                    mean, _, x_start = _grad_ckpt(
                        self._compute_transition_params,
                        model, context, key_padding_mask, target_indices, x_t, times,
                        use_reentrant=False,
                    )
                else:
                    mean, _, x_start = self._compute_transition_params(
                        model, context, key_padding_mask, target_indices, x_t, times
                    )

                if t == 0:
                    x_prev = x_start
                else:
                    _, _, log_var = model.q_posterior(x_start, x_t, times)
                    x_prev = mean + torch.exp(0.5 * log_var) * torch.randn_like(x_t)

                transitions.append(
                    {
                        "context": context.detach(),
                        "key_padding_mask": key_padding_mask.detach(),
                        "target_indices": target_indices.detach(),
                        "times": times.detach(),
                        "x_t": x_t.detach(),
                    }
                )
                x_t = x_prev

            generated_states.append(x_t)
            seq_context[:, pos] = x_t.detach()

        seq = torch.stack(generated_states, dim=1)
        return seq, transitions

    def _accumulate_kl_grads(self, transitions: Iterable[Dict[str, torch.Tensor]], loss_scale: float = 1.0) -> float:
        """Per-chunk KL backward: accumulate grads chunk-by-chunk and free each graph.

        Mathematically identical to summing chunk KLs then a single backward, but the
        graph for a chunk is released after its backward, so peak memory is one chunk
        instead of all transitions at once. Returns the unweighted total KL (float) for
        logging; gradients are already scaled by (loss_scale * self.kl_weight).
        """
        transitions = list(transitions)
        if len(transitions) == 0:
            raise ValueError("No transitions collected from rollout")

        chunk_size = self.transition_chunk_size if self.transition_chunk_size > 0 else len(transitions)
        total_kl = 0.0

        for start_idx in range(0, len(transitions), chunk_size):
            chunk = transitions[start_idx:start_idx + chunk_size]
            context = torch.cat([tr["context"] for tr in chunk], dim=0)
            key_padding_mask = torch.cat([tr["key_padding_mask"] for tr in chunk], dim=0)
            target_indices = torch.cat([tr["target_indices"] for tr in chunk], dim=0)
            times = torch.cat([tr["times"] for tr in chunk], dim=0)
            x_t = torch.cat([tr["x_t"] for tr in chunk], dim=0)

            _, pred_noise_new, _ = self._compute_transition_params(
                self.diffusion,
                context,
                key_padding_mask,
                target_indices,
                x_t,
                times,
            )

            with torch.no_grad():
                _, pred_noise_ref, _ = self._compute_transition_params(
                    self.reference,
                    context,
                    key_padding_mask,
                    target_indices,
                    x_t,
                    times,
                )

            kl = (pred_noise_new - pred_noise_ref).pow(2).flatten(start_dim=1).mean(dim=-1).sum()
            (loss_scale * self.kl_weight * kl).backward()
            total_kl += float(kl.detach().item())

        return total_kl

    def step(self, batch_size: int, reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
             *, zero_grad: bool = True, do_step: bool = True, loss_scale: float = 1.0) -> FineTuneStats:
        """One rollout + backward.

        Reward and KL backward are done separately so their (independent) graphs are
        released as soon as each is consumed. For gradient accumulation, call with
        zero_grad only on the first microstep, do_step only on the last, and
        loss_scale=1/num_microsteps; grads accumulate across microsteps and the
        optimizer/clip fire once at the end.
        """
        reward_fn = reward_fn or self.reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn is required. Plug your sequence reward API via constructor or step(...).")

        # Keep policy/reference in eval mode during rollout + KL computation to avoid dropout-induced KL noise.
        self.diffusion.eval()
        self.reference.eval()
        samples, transitions = self.rollout(batch_size=batch_size, show_progress=False)

        rewards = reward_fn(samples)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards = rewards.to(self.device).float()
        if rewards.ndim != 1 or rewards.shape[0] != batch_size:
            raise ValueError(f"reward_fn must return shape [batch_size], got {tuple(rewards.shape)}")

        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)

        # Reward term: backward through the (checkpointed) rollout chain, then free it.
        reward_loss = -rewards.mean()
        (loss_scale * reward_loss).backward()

        # KL term: per-chunk backward (grads already scaled inside).
        kl_value = self._accumulate_kl_grads(transitions, loss_scale=loss_scale)

        report_loss = float(reward_loss.item()) + self.kl_weight * kl_value

        if do_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.diffusion.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            self.optimizer.step()
        else:
            grad_norm_value = 0.0

        with torch.no_grad():
            return FineTuneStats(
                loss=float(report_loss),
                policy_loss=float(reward_loss.item()),
                kl_loss=float(kl_value),
                reward_mean=float(rewards.mean().item()),
                reward_std=float(rewards.std(unbiased=False).item()),
                grad_norm=float(grad_norm_value),
            )
