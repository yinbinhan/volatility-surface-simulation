import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


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




class ArbitrageValidator:
    """Arbitrage-violation calculator for generated volatility/price surfaces."""

    def __init__(self, grid_m, grid_t):
        self.m = torch.tensor(grid_m, dtype=torch.float32)
        self.t = torch.tensor(grid_t, dtype=torch.float32)
        self.dt = self.t[1:] - self.t[:-1]
        self.dm = self.m[1:] - self.m[:-1]

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
        batch_size, seq_len = prices.shape[:2]
        prices = prices[..., 1, :, :].reshape(batch_size * seq_len, len(t), len(self.m))

        diff_t = prices[:, :-1, :] - prices[:, 1:, :]
        l1_map = torch.relu(t[:-1].view(1, -1, 1) * diff_t / dt.view(1, -1, 1))
        l1 = torch.sum(l1_map, dim=(1, 2))

        diff_m = prices[:, :, 1:] - prices[:, :, :-1]
        slope = diff_m / dm.view(1, 1, -1)
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


class OnlineDDPMLoRAFineTuner:
    """Online policy-gradient fine-tuning for sequential DDPM with LoRA.

    - Generates fresh rollouts each step (no replay buffer).
    - Uses sequence-level rewards supplied by an external callback.
    - Regularizes policy against a frozen reference with closed-form Gaussian KL.
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
        device: Optional[torch.device] = None,
    ):
        self.diffusion = diffusion
        self.device = device or diffusion.device
        self.reward_fn = reward_fn
        self.kl_weight = kl_weight
        self.normalize_rewards = normalize_rewards

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
        mean, _, log_var = model.q_posterior(x_start, x_t, times)
        return mean, log_var, pred_noise, x_start

    @torch.no_grad()
    def rollout(self, batch_size: int, show_progress: bool = False):
        model = self.diffusion
        seq = torch.zeros((batch_size, model.seq_len, *model.state_shape), device=self.device)
        transitions: List[Dict[str, torch.Tensor]] = []

        positions = range(model.seq_len)
        if show_progress:
            positions = tqdm(positions, desc="FT rollout positions", leave=False)

        for pos in positions:
            x_t = torch.randn((batch_size, *model.state_shape), device=self.device)
            target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)

            for t in reversed(range(model.num_timesteps)):
                times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                context, key_padding_mask = model.build_context(seq, target_indices, x_t)

                mean, log_var, _, x_start = self._compute_transition_params(
                    model, context, key_padding_mask, target_indices, x_t, times
                )

                if t == 0:
                    x_prev = x_start
                else:
                    x_prev = mean + torch.exp(0.5 * log_var) * torch.randn_like(x_t)

                transitions.append(
                    {
                        "context": context.detach(),
                        "key_padding_mask": key_padding_mask.detach(),
                        "target_indices": target_indices.detach(),
                        "times": times.detach(),
                        "x_t": x_t.detach(),
                        "x_prev": x_prev.detach(),
                    }
                )
                x_t = x_prev

            seq[:, pos] = x_t

        return seq, transitions

    def _compute_logprob_and_kl(self, transitions: Iterable[Dict[str, torch.Tensor]]):
        total_log_prob = None
        total_kl = None

        for tr in transitions:
            mean_new, log_var_new, _, _ = self._compute_transition_params(
                self.diffusion,
                tr["context"],
                tr["key_padding_mask"],
                tr["target_indices"],
                tr["x_t"],
                tr["times"],
            )
            log_prob = gaussian_log_prob(tr["x_prev"], mean_new, log_var_new)

            with torch.no_grad():
                mean_ref, log_var_ref, _, _ = self._compute_transition_params(
                    self.reference,
                    tr["context"],
                    tr["key_padding_mask"],
                    tr["target_indices"],
                    tr["x_t"],
                    tr["times"],
                )
            kl = gaussian_kl_divergence(mean_new, log_var_new, mean_ref, log_var_ref)

            if total_log_prob is None:
                total_log_prob = log_prob
                total_kl = kl
            else:
                total_log_prob = total_log_prob + log_prob
                total_kl = total_kl + kl

        return total_log_prob, total_kl

    def step(self, batch_size: int, reward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> FineTuneStats:
        reward_fn = reward_fn or self.reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn is required. Plug your sequence reward API via constructor or step(...).")

        self.diffusion.train()
        samples, transitions = self.rollout(batch_size=batch_size, show_progress=False)

        rewards = reward_fn(samples)
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        rewards = rewards.to(self.device).float()
        if rewards.ndim != 1 or rewards.shape[0] != batch_size:
            raise ValueError(f"reward_fn must return shape [batch_size], got {tuple(rewards.shape)}")

        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        log_prob_sum, kl_sum = self._compute_logprob_and_kl(transitions)

        policy_loss = -(rewards.detach() * log_prob_sum).mean()
        kl_loss = kl_sum.mean()
        loss = policy_loss + self.kl_weight * kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            return FineTuneStats(
                loss=float(loss.item()),
                policy_loss=float(policy_loss.item()),
                kl_loss=float(kl_loss.item()),
                reward_mean=float(rewards.mean().item()),
                reward_std=float(rewards.std(unbiased=False).item()),
            )
