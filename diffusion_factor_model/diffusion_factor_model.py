import math
import copy
import warnings
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from scipy.stats import ortho_group
from torch.utils.tensorboard import SummaryWriter
import os

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from torch.optim import AdamW
from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from diffusion_factor_model.attend import Attend

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        filter_size = 7,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, filter_size, padding = int((filter_size-1)/2))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class ConditionalTransformer(Module):
    """Transformer that predicts the noise of one entry conditioned on history."""

    def __init__(
        self,
        *,
        seq_len,
        dim=256,
        depth=6,
        heads=8,
        ff_mult=4,
        dropout=0.1,
        use_bos_token=False,
        use_alibi=False,
        alibi_slope=1.0,
        first_token_bias=0.0,
        state_shape=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_bos_token = use_bos_token
        self.use_alibi = use_alibi
        self.alibi_slope = alibi_slope
        self.first_token_bias = first_token_bias
        self.state_shape = tuple(state_shape) if state_shape is not None else ()
        self.state_dim = int(np.prod(self.state_shape)) if len(self.state_shape) > 0 else 1
        self.value_proj = nn.Linear(self.state_dim, dim)
        self.indicator_embed = nn.Embedding(2, dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        emb_len = seq_len + (1 if use_bos_token else 0)
        self.pos_emb = nn.Parameter(torch.randn(emb_len, dim))
        if use_bos_token:
            self.bos_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ff_mult),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        mask = torch.triu(torch.ones(emb_len, emb_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('causal_mask', mask)
        if self.use_alibi or self.first_token_bias != 0.0:
            positions = torch.arange(emb_len)
            distances = positions.unsqueeze(0) - positions.unsqueeze(1)
            distances = distances.clamp_min(0).float()
            alibi_bias = torch.zeros(emb_len, emb_len)
            if self.use_alibi:
                alibi_bias -= self.alibi_slope * distances
            if self.first_token_bias != 0.0:
                alibi_bias[:, 0] += self.first_token_bias
            self.register_buffer('alibi_bias', alibi_bias)
        self.dropout = nn.Dropout(dropout)
        # Small prediction head so representation and scalar output can live in different spaces
        # Empirically, stacking two hidden layers stabilizes gradients, so keep both.
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, self.state_dim)
        )

    def _flatten_states(self, values):
        if values.ndim < 2:
            raise ValueError('values must have shape [batch, seq_len, ...]')
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        batch, seq_len = values.shape[:2]
        values = values.reshape(batch, seq_len, -1)
        if values.shape[-1] != self.state_dim:
            raise ValueError(
                f'expected per-step state dimension {self.state_dim}, got {values.shape[-1]}. '
                'Set `state_shape` to match your data.'
            )
        return values

    def forward(self, values, target_indices, timesteps, key_padding_mask):
        """Forward pass.

        Args:
            values: Tensor of shape [batch, seq_len, ...] containing history and noisy entry.
            target_indices: Long tensor [batch] for the index of the noisy entry.
            timesteps: Long tensor [batch] for diffusion step.
            key_padding_mask: Bool tensor [batch, seq_len] masking padded entries (True => ignore).
        Returns:
            Predicted noise tensor for each batch element with shape [batch, *state_shape]
            (or [batch] when state_shape is empty).
        """
        values = self._flatten_states(values)
        batch, seq_len, _ = values.shape
        device = values.device
        tokens = self.value_proj(values)
        offset = 1 if self.use_bos_token else 0
        if self.use_bos_token:
            bos = self.bos_token.expand(batch, -1, -1)
            tokens = torch.cat([bos, tokens], dim=1)

        seq_len_tokens = tokens.shape[1]
        pos = self.pos_emb[:seq_len_tokens].unsqueeze(0)
        tokens = tokens + pos
        indicator = torch.zeros(batch, seq_len_tokens, dtype=torch.long, device=device)
        indicator.scatter_(1, (target_indices + offset).unsqueeze(1), 1)
        tokens = tokens + self.indicator_embed(indicator)
        time_emb = self.time_mlp(timesteps.float())
        tokens[torch.arange(batch, device=device), target_indices + offset] += time_emb
        tokens = self.dropout(tokens)
        mask = self.causal_mask[:seq_len_tokens, :seq_len_tokens]
        if self.use_alibi or self.first_token_bias != 0.0:
            mask = mask + self.alibi_bias[:seq_len_tokens, :seq_len_tokens]
        if self.use_bos_token:
            key_padding_mask = F.pad(key_padding_mask, (1, 0), value=False)
        encoded = self.encoder(tokens, mask=mask, src_key_padding_mask=key_padding_mask)
        target_states = encoded[torch.arange(batch, device=device), target_indices + offset]
        pred = self.output(target_states)
        if len(self.state_shape) == 0:
            return pred.squeeze(-1)
        return pred.view(batch, *self.state_shape)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 10):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
    
# Numerical Experiments --- generator of our simulated data    

class GaussianLatentSampler2D_Finance(object):
    def __init__(self, d_inner, image_size):
        self.image_size = image_size
        self.d_inner, self.d_outer = d_inner, image_size[0]*image_size[1]
        self.A = np.random.randn(self.d_inner, self.d_outer)

    def generate_data(self, N, latent_mean, latent_cov, noise_mean=None, noise_cov=None, sort_var=True, torch_tensor=False):
        if sort_var:
            if noise_cov is not None:
                diagonal_elements = np.diag(self.A.T.dot(latent_cov).dot(self.A) + noise_cov)
                # Sort the diagonal elements in descending order and get the indices
                sorted_indices = np.argsort(diagonal_elements)[::-1]
                factor = np.random.multivariate_normal(latent_mean, latent_cov, N).dot(self.A)
                x = factor + np.random.multivariate_normal(noise_mean, noise_cov, N)
                factor = factor[:, sorted_indices]
                x = x[:, sorted_indices]
            else:
                diagonal_elements = np.diag(self.A.T.dot(latent_cov).dot(self.A))
                # Sort the diagonal elements in descending order and get the indices
                sorted_indices = np.argsort(diagonal_elements)[::-1]
                factor = np.random.multivariate_normal(latent_mean, latent_cov, N).dot(self.A)
                x = factor
                factor = factor[:, sorted_indices]
                x = x[:, sorted_indices]
        else:
            if noise_cov is not None:
                factor = np.random.multivariate_normal(latent_mean, latent_cov, N).dot(self.A)
                x = factor + np.random.multivariate_normal(noise_mean, noise_cov, N)
            else:
                factor = np.random.multivariate_normal(latent_mean, latent_cov, N).dot(self.A)
                x = factor
        if torch_tensor:
            factor = torch.from_numpy(factor).float()
            x = torch.from_numpy(x).float()

        return factor, x.reshape((N, self.image_size[0], self.image_size[1]))

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        latent_dim = 5,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.in_channels
        self.self_condition = False

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.objective = objective
        self.features = image_size[0] * image_size[1]

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device
    
    def noise_pos(self, x, i):
        B, L, device = *x.shape, x.device
        M = torch.arange(L, device=device).unsqueeze(0) == i.unsqueeze(1)         # [B, L]
        M = M.float()

        idx = torch.arange(L, device=device).unsqueeze(0)
        causal_mask = (idx <= i.unsqueeze(1)).float()

        return M, causal_mask

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, i, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        # Determine noise injection positions
        M = torch.zeros(x.shape, device = self.device)
        M[:, i] = torch.ones(M[:, i].shape, device = self.device)
        
        # reshape x before feeding into DiT
        h, w = self.image_size
        x = x.reshape(-1, 1, h, w)
        
        model_output = self.model(x, t, i, None)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # reshape x back and zero out entries other than i
        x = x.view(model_output.size(0), -1) * M
        

        if self.objective == 'pred_noise':
            pred_noise = model_output.view(model_output.size(0), -1) * M
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output.view(model_output.size(0), -1) * M
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, i, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(x, t, i, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, i: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_t = torch.full((b,), t, device = device, dtype = torch.long)
        batched_i = torch.full((b,), i, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_t, i = batched_i, x_self_cond = x_self_cond, clip_denoised = False)
        model_mean = model_mean[:, i].reshape(-1, 1)
        noise = torch.randn_like(model_mean) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        pred_img = pred_img.squeeze(1) 
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, save_timesteps = None, return_all_timesteps = False, pbar = None):
        """
        Perform the reverse diffusion sampling loop.
        
        Args:
            shape: Shape of the images to generate (batch_size, channels, height, width)
            save_timesteps: List/set of specific timesteps to save for early stopping evaluation. 
                          If provided, only saves samples at these exact timesteps (no initial noise).
                          Useful for evaluating model quality at intermediate denoising steps.
            return_all_timesteps: If True, returns samples from all timesteps (for visualization/debugging)
        
        Returns:
            Generated samples. Shape depends on settings:
            - If save_timesteps: (batch, num_specified_timesteps, channels, height, width)
            - If return_all_timesteps: (batch, num_timesteps+1, channels, height, width) - includes initial noise
            - Otherwise: (batch, channels, height, width)
        """
        batch, device = shape[0], self.device

        img = torch.zeros(shape, device = device)
        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        if return_all_timesteps:
            # Save samples from all timesteps for visualization or analysis
            imgs = [img]  # Include initial noise
            for t in reversed(range(0, self.num_timesteps)):
                self_cond = x_start if self.self_condition else None
                img, x_start = self.p_sample(img, t, self_cond)
                imgs.append(img)
                
            ret = torch.stack(imgs, dim = 1)
            
        elif exists(save_timesteps):
            # Save only specified timesteps for early stopping evaluation
            # Only saves the exact timesteps requested, no initial noise
            # imgs = []
            for i in range(0, self.features):
                img[:, i] = torch.randn(img[:, i].shape, device = device)
                for t in reversed(range(0, self.num_timesteps)):
                    self_cond = x_start if self.self_condition else None
                    img[:, i], x_start = self.p_sample(img, t, i, self_cond)
                    if t in save_timesteps:
                        # imgs.append(img)
                        if pbar is not None:
                            pbar.update(1)   
                        break
                    
                    
            # ret = torch.stack(imgs, dim = 1)
            ret = img
        
        else:
            # Default: only return the final fully denoised sample
            for t in reversed(range(0, self.num_timesteps)):
                self_cond = x_start if self.self_condition else None
                img, x_start = self.p_sample(img, t, self_cond)
            
            ret = img

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, save_timesteps = None, return_all_timesteps = False):
        """
        Perform DDIM (Denoising Diffusion Implicit Models) sampling for faster generation.
        
        Args:
            shape: Shape of the images to generate (batch_size, channels, height, width)
            save_timesteps: Optional list/set of specific timesteps to save for early stopping evaluation.
                          Only saves samples at these exact timesteps (no initial noise).
                          Example: [100, 200, 500] will save samples at only these three timesteps.
            return_all_timesteps: If True, returns samples from all sampling timesteps (for analysis/debugging).
                                 Takes precedence over save_timesteps if both are provided.
        
        Returns:
            Generated samples with shape:
            - If save_timesteps: (batch_size, num_specified_timesteps, channels, height, width)
            - If return_all_timesteps: (batch_size, num_sampling_steps+1, channels, height, width) - includes initial noise
            - Otherwise: (batch_size, channels, height, width)
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        # Determine whether to save all timesteps or specific timesteps for early stopping
        if return_all_timesteps:
            # Save all sampling timesteps including initial noise
            imgs = [img]
            #for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise

                imgs.append(img)
                
            ret = torch.stack(imgs, dim = 1)
            
        elif exists(save_timesteps):
            # Save only specified timesteps for early stopping evaluation
            # Only saves the exact timesteps requested, no initial noise
            imgs = []
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    if time in save_timesteps:
                        imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise

                if time in save_timesteps:
                    imgs.append(img)
                    
            ret = torch.stack(imgs, dim = 1)
        
        else:
            # Default: only return the final denoised sample
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

                if time_next < 0:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
            
            ret = img

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, save_timesteps = None, return_all_timesteps = False, pbar = None):
        """
        Generate samples from the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            save_timesteps: Optional list/set of specific timesteps to save during the reverse diffusion process.
                          This is useful for early stopping evaluation - you can specify timesteps like [100, 200, 500]
                          to save intermediate denoising results at only those exact timesteps.
                          No initial noise is saved, only the specified timesteps.
                          If None, only returns the final fully denoised samples.
            return_all_timesteps: If True, returns samples from all timesteps including initial noise (for visualization/debugging).
                                 Takes precedence over save_timesteps if both are provided.
        
        Returns:
            Generated samples with shape:
            - (batch_size, channels, height, width) if save_timesteps is None and return_all_timesteps is False
            - (batch_size, len(save_timesteps), channels, height, width) if save_timesteps is provided
            - (batch_size, num_timesteps+1, channels, height, width) if return_all_timesteps is True
        """
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, self.features), save_timesteps = save_timesteps, return_all_timesteps = return_all_timesteps, pbar = pbar)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, M, causal_mask, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]


        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape) 
        sqrt_oma   = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Only modify position i: elsewhere keep x_start unchanged
        x_t = x_start + M * ((sqrt_alpha - 1.0) * x_start + sqrt_oma * noise)

        # now zero out all positions after i  (enforcing causality)
        
        # x_t = x_t * causal_mask

        return x_t
    
    


    def p_losses(self, x_start, t, i, noise = None, offset_noise_strength = None):
        # b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))
        M, causal_mask = self.noise_pos(x_start, i)

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, M = M, causal_mask = causal_mask, noise = noise)


        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # # reshape before feeding into model
        # h, w = self.image_size
        # x = x.reshape(-1, 1, h, w)

        # predict and take gradient step
        model_out = self.model(x, t, i, x_self_cond)
        # model_out = model_out.view(model_out.size(0), -1)
        # model_out = model_out * M

        if self.objective == 'pred_noise':
            target = noise[M == 1].reshape(-1, 1)
        elif self.objective == 'pred_x0':
            target = x_start[M == 1].reshape(-1, 1)
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, _, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        i = torch.randint(0, self.features, (b,), device=device).long() # i for index of time series

        img = self.normalize(img)
        return self.p_losses(img, t, i, *args, **kwargs)

class SequentialGaussianDiffusion(Module):
    """Sequential diffusion process that denoises entries one-by-one."""

    def __init__(
        self,
        model,
        *,
        seq_len,
        timesteps=1000,
        sampling_timesteps=None,
        ddim_eta=0.0,
        objective='pred_noise',
        beta_schedule='cosine',
        schedule_fn_kwargs=dict(),
        auto_normalize=False,
        state_shape=None,
    ):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.objective = objective
        inferred_state_shape = getattr(model, 'state_shape', ())
        self.state_shape = tuple(state_shape) if state_shape is not None else tuple(inferred_state_shape)

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = int(betas.shape[0])

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
        else:
            raise ValueError(f'unknown objective {objective}')
        register_buffer('loss_weight', loss_weight)

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.sampling_timesteps = self.num_timesteps if sampling_timesteps is None else int(sampling_timesteps)
        if self.sampling_timesteps <= 0:
            raise ValueError("sampling_timesteps must be a positive integer")
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = float(ddim_eta)

    @property
    def device(self):
        return self.betas.device

    def build_context(self, sequences, target_indices, target_values):
        device = sequences.device
        batch, seq_len = sequences.shape[:2]
        arange = torch.arange(seq_len, device=device)
        prefix_mask = arange.unsqueeze(0) < target_indices.unsqueeze(1)
        while prefix_mask.ndim < sequences.ndim:
            prefix_mask = prefix_mask.unsqueeze(-1)
        context = torch.zeros_like(sequences)
        context = torch.where(prefix_mask, sequences, context)
        context[torch.arange(batch, device=device), target_indices] = target_values
        key_padding_mask = arange.unsqueeze(0) > target_indices.unsqueeze(1)
        return context, key_padding_mask

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_noise_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, sequences, t, target_indices, noise=None):
        batch = sequences.shape[0]
        target = sequences[torch.arange(batch, device=self.device), target_indices]
        noise = default(noise, lambda: torch.randn_like(target))
        noisy_target = self.q_sample(target, t, noise)
        context, key_padding_mask = self.build_context(sequences, target_indices, noisy_target)
        pred = self.model(context, target_indices, t, key_padding_mask)

        if self.objective == 'pred_noise':
            target_value = noise
        elif self.objective == 'pred_x0':
            target_value = target
        elif self.objective == 'pred_v':
            target_value = self.predict_v(target, t, noise)
        loss = F.mse_loss(pred, target_value, reduction='none')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sequences, *args, **kwargs):
        batch, seq_len = sequences.shape[:2]
        assert seq_len == self.seq_len, f'sequence length must be {self.seq_len}'
        expected_ndim = 2 + len(self.state_shape)
        if sequences.ndim != expected_ndim:
            raise ValueError(
                f'expected input shape [batch, seq_len, *state_shape] with {expected_ndim} dimensions, '
                f'got {sequences.shape}'
            )
        if len(self.state_shape) > 0 and tuple(sequences.shape[2:]) != self.state_shape:
            raise ValueError(
                f'expected trailing shape {self.state_shape}, got {tuple(sequences.shape[2:])}'
            )
        t = torch.randint(0, self.num_timesteps, (batch,), device=sequences.device).long()
        target_indices = torch.randint(0, self.seq_len, (batch,), device=sequences.device).long()
        return self.p_losses(sequences, t, target_indices, *args, **kwargs)

    def _build_ddim_time_pairs(self):
        device = self.device
        times = torch.linspace(0, self.num_timesteps - 1, steps=self.sampling_timesteps, device=device)
        times = torch.flip(times.long(), dims=[0]).tolist()
        if len(times) == 1:
            return [(times[0], -1)]
        return list(zip(times[:-1], times[1:] + [-1]))

    def _model_predictions(self, context, key_padding_mask, target_indices, x_t, times):
        """Return predicted noise and x_start for a given timestep.

        Supports objectives of predicting noise, x0, or velocity (v).
        """

        model_out = self.model(context, target_indices, times, key_padding_mask)

        if self.objective == 'pred_noise':
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x_t, times, pred_noise)
        elif self.objective == 'pred_x0':
            x_start = model_out
            pred_noise = self.predict_noise_from_start(x_t, times, x_start)
        elif self.objective == 'pred_v':
            x_start = self.predict_start_from_v(x_t, times, model_out)
            pred_noise = self.predict_noise_from_v(x_t, times, model_out)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return pred_noise, x_start

    def _ddpm_step(self, sequences, pos):
        batch_size = sequences.size(0)
        x_t = torch.randn((batch_size, *self.state_shape), device=self.device)
        target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)
        for t in reversed(range(self.num_timesteps)):
            times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            context, key_padding_mask = self.build_context(sequences, target_indices, x_t)
            pred_noise, x_start = self._model_predictions(
                context,
                key_padding_mask,
                target_indices,
                x_t,
                times,
            )
            if t == 0:
                x_t = x_start
            else:
                model_mean, _, log_variance = self.q_posterior(x_start, x_t, times)
                noise = torch.randn_like(x_t)
                x_t = model_mean + (0.5 * log_variance).exp() * noise
        return x_t

    def _ddim_step(self, sequences, pos):
        batch_size = sequences.size(0)
        x_t = torch.randn((batch_size, *self.state_shape), device=self.device)
        target_indices = torch.full((batch_size,), pos, device=self.device, dtype=torch.long)
        time_pairs = self._build_ddim_time_pairs()

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            context, key_padding_mask = self.build_context(sequences, target_indices, x_t)
            pred_noise, x_start = self._model_predictions(
                context,
                key_padding_mask,
                target_indices,
                x_t,
                time_cond,
            )

            if time_next < 0:
                x_t = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x_t

    @torch.inference_mode()
    def sample(
        self,
        batch_size=16,
        save_timesteps=None,
        return_all_timesteps=False,
        conditioning=None,
        conditioning_mask=None,
        start_idx=0,
        end_idx=None,
        show_progress=False,
        progress_desc=None,
    ):
        if save_timesteps is not None:
            warnings.warn(
                "Sequential sampling currently ignores `save_timesteps`; only the fully denoised sequences are returned.",
                stacklevel=2,
            )
        if return_all_timesteps:
            warnings.warn(
                "Sequential sampling does not expose intermediate histories; returning only the final sequences.",
                stacklevel=2,
            )

        sequences = torch.zeros((batch_size, self.seq_len, *self.state_shape), device=self.device)
        if conditioning is not None:
            conditioning = conditioning.to(self.device)
            if conditioning.dim() == 1 + len(self.state_shape):
                conditioning = conditioning.unsqueeze(0)
            if conditioning.shape != sequences.shape:
                raise ValueError(
                    "conditioning must have shape [batch_size, seq_len, *state_shape] or [seq_len, *state_shape]"
                )
            if conditioning_mask is None:
                raise ValueError("conditioning_mask is required when conditioning is provided")
            if conditioning_mask.dim() == 1:
                conditioning_mask = conditioning_mask.unsqueeze(0).expand(batch_size, -1)
            if conditioning_mask.shape != sequences.shape[:2]:
                raise ValueError("conditioning_mask must have shape [batch_size, seq_len] or [seq_len]")
            conditioning_mask = conditioning_mask.bool()
            if not (conditioning_mask == conditioning_mask[0]).all():
                raise ValueError("conditioning_mask must be identical across the batch")
            mask = conditioning_mask
            while mask.ndim < sequences.ndim:
                mask = mask.unsqueeze(-1)
            sequences = torch.where(mask, conditioning, sequences)
        elif conditioning_mask is not None:
            raise ValueError("conditioning_mask provided without conditioning values")
        sample_pos = self._ddim_step if self.is_ddim_sampling else self._ddpm_step

        if end_idx is None:
            end_idx = self.seq_len
        start_idx = max(0, int(start_idx))
        end_idx = min(self.seq_len, int(end_idx))
        if start_idx >= end_idx:
            raise ValueError("Sampling window must include at least one index")

        if conditioning_mask is not None:
            conditioned_positions = conditioning_mask[0].bool()
            indices_to_generate = [
                pos
                for pos in range(start_idx, end_idx)
                if not conditioned_positions[pos].item()
            ]
        else:
            indices_to_generate = list(range(start_idx, end_idx))
        iterator = indices_to_generate
        if show_progress:
            iterator = tqdm(
                indices_to_generate,
                desc=progress_desc or "Sampling",
                unit="index",
                leave=False,
            )

        for pos in iterator:
            sequences[:, pos] = sample_pos(sequences, pos)
            if show_progress:
                iterator.set_postfix(index=pos)

        return sequences

class WarmUpCosineAnnealingWarmRestarts:
    def __init__(self, optimizer, warmup_iters=10, T_0=100, T_mult=1, eta_min=1e-6, cosine_steps=200, last_epoch=-1):
        """
        A learning rate scheduler that combines linear warmup, cosine annealing with restarts,
        and a constant learning rate phase at the end.

        Args:
            optimizer: The optimizer to which the scheduler will be applied.
            warmup_iters: Number of warmup iterations.
            T_0: Initial number of iterations for the first cosine cycle.
            T_mult: Multiplicative factor for the length of subsequent cycles.
            eta_min: Minimum learning rate for cosine annealing.
            constant_steps: Number of steps to keep the learning rate constant after warmup and cosine annealing.
            last_epoch: The index of the last epoch (default: -1).
        """
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.cosine_steps = cosine_steps

        # Linear warmup function
        def warmup_fn(step):
            if step < self.warmup_iters:
                return float(step) / float(max(1, self.warmup_iters))
            return 1.0

        # Warmup scheduler
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_fn)
        # Cosine annealing with restarts scheduler
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min, last_epoch=self.last_epoch
        )

    def step(self, epoch=None):
        """
        Update the learning rate.

        Args:
            epoch: The current epoch (default: None).
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch < self.warmup_iters:
            # Warmup phase
            self.warmup_scheduler.step(epoch)
        elif epoch < self.warmup_iters + self.cosine_steps:
            # Cosine annealing phase
            self.cosine_scheduler.step(epoch - self.warmup_iters)
        else:
            # Constant learning rate phase
            pass  # Do nothing, keep the learning rate constant
        
class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        optimizer=None,
        scheduler=None,
        *,
        train_lr=1e-4,
        adamw_weight_decay=0.01,
        cosine_scheduler=True,
        warm_up=True,
        warmup_iters=10,
        T_0=40,
        T_mult=1,
        eta_min=1e-6,
        cosine_steps=None,
        train_batch_size=32,
        gradient_accumulate_every=1,
        train_epochs=10,
        ema_update_every=4,
        ema_decay=0.995,
        save_and_sample_every=10,
        num_samples=256,
        results_folder='./results',
        param_path='',
        amp=False,
        mixed_precision_type='fp16',
        split_batches=False,
        max_grad_norm=1.,
        num_workers=0,
        calculate_fid=False,
        num_fid_samples=0,
        save_best_and_latest_only=False,
        save_timesteps=None  # Specific timesteps to save for early stopping evaluation
    ):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize accelerator for mixed precision training
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            device_placement=True, cpu=False
        )

        # Model and optimizer, scheduler
        self.model = diffusion_model
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.99), weight_decay=adamw_weight_decay)
        else:
            self.optimizer = optimizer
        
        # Create scheduler if not provided
        if scheduler is None:
            if cosine_scheduler and warm_up:
                # Use warm-up + cosine annealing with restarts
                # If cosine_steps not provided, use T_0 as fallback
                if cosine_steps is None:
                    cosine_steps = T_0
                self.scheduler = WarmUpCosineAnnealingWarmRestarts(
                    self.optimizer, 
                    warmup_iters=warmup_iters,
                    T_0=T_0,
                    T_mult=T_mult,
                    eta_min=eta_min,
                    cosine_steps=cosine_steps
                )
            elif cosine_scheduler:
                # Use cosine annealing without warm-up
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_0,
                    eta_min=eta_min
                )
            else:
                # Use constant learning rate (identity scheduler)
                self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            self.scheduler = scheduler

        # Dataloader
        self.dataloader = DataLoader(dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=num_workers)
        self.dataloader = self.accelerator.prepare(self.dataloader)

        # Training hyperparameters
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_epochs = train_epochs
        self.max_grad_norm = max_grad_norm
        self.save_and_sample_every = save_and_sample_every
        self.num_samples = num_samples
        self.num_fid_samples = num_fid_samples
        self.save_timesteps = save_timesteps  # Store timesteps for early stopping evaluation

        # Model checkpoint and result saving
        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Results folder setup
        self.checkpoint_folder = self.results_folder / param_path
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        # Exponential Moving Average (EMA)
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # Prepare model, optimizer, and scheduler with accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        # Step counter
        self.step = 0

        # FID calculation (optional)
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process
        if self.calculate_fid:
            from fid_evaluation import FIDEvaluation  # Assume FID module is available
            self.fid_scorer = FIDEvaluation(
                batch_size=train_batch_size,
                dl=self.dataloader,
                sampler=self.ema.ema_model,
                channels=self.model.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                num_fid_samples=num_fid_samples
            )
            self.best_fid = float("inf") if save_best_and_latest_only else None

        self.save_best_and_latest_only = save_best_and_latest_only

        # TensorBoard logger
        self.logger = SummaryWriter(log_dir=self.checkpoint_folder)

    def train(self):
        device = self.device
        self.model.to(device)

        for epoch in range(self.train_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            total_batches = len(self.dataloader)
            update_pbar_batches = total_batches  # // 2
            grad_norm_total = 0.0
            grad_norm_count = 0
            # Update scheduler for each epoch
            self.scheduler.step(epoch)
            
            with tqdm(total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.train_epochs}", disable=not self.accelerator.is_main_process) as pbar:
                for batch_idx, data in enumerate(self.dataloader):
                    data = data[0].to(device)

                    # Forward pass with gradient accumulation
                    with self.accelerator.autocast():
                        batch_loss = self.model(data) / self.gradient_accumulate_every
                        total_loss += batch_loss.item()

                    self.accelerator.backward(batch_loss)
                    num_batches += 1

                    # Update model parameters after accumulating `gradient_accumulate_every` batches
                    should_step = (batch_idx + 1) % self.gradient_accumulate_every == 0
                    is_last_batch = (batch_idx + 1) == total_batches
                    if should_step or is_last_batch:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                        if math.isfinite(grad_norm_value):
                            grad_norm_total += grad_norm_value
                            grad_norm_count += 1
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Update EMA if required
                        if self.accelerator.is_main_process and self.ema and (self.step + 1) % self.ema.update_every == 0:
                            self.ema.update()

                    self.step += 1

                    # Update progress bar and display loss only when update_pbar_batches is reached
                    if (batch_idx + 1) % update_pbar_batches == 0:
                        pbar.set_postfix(loss=total_loss / num_batches)
                        pbar.update(update_pbar_batches)

                # Log metrics at the end of the epoch
                avg_train_loss = total_loss / num_batches
                avg_grad_norm = grad_norm_total / grad_norm_count if grad_norm_count else 0.0

                self.logger.add_scalar('Train/Average Loss', avg_train_loss, epoch)
                self.logger.flush()

                self.accelerator.print(
                    f"Epoch {epoch + 1}/{self.train_epochs} completed with avg loss {avg_train_loss:.4f} "
                    f"and avg grad norm {avg_grad_norm:.4f}"
                )

                # Save model and generate samples at the end of each epoch
                if self.accelerator.is_main_process and (epoch + 1) % self.save_and_sample_every == 0:
                    self.save_and_sample(epoch)  # Save samples and checkpoint at each epoch

        self.logger.close()
        self.accelerator.print("Training complete")

    def save_checkpoint(self, epoch):
        # Save checkpoint at the end of each epoch

        checkpoint_path = self.checkpoint_folder / f"model-epoch-{epoch+1}.pt"
        
        checkpoint = {
            "epoch": epoch + 1,
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None
        }
        torch.save(checkpoint, checkpoint_path)
        
    def load(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        # Load checkpoint file
        data = torch.load(path, map_location=device, weights_only=True)

        # Load parameters
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.optimizer.load_state_dict(data['optimizer'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")
    
    def save_and_sample(self, epoch):
        # Save samples and model checkpoint at each epoch
        if self.num_fid_samples > 0:
            if self.ema:
                self.ema.ema_model.eval()
                with torch.no_grad():
                    # Pass save_timesteps parameter for early stopping evaluation
                    samples = self.ema.ema_model.sample(self.num_new_samples, save_timesteps=self.save_timesteps)
            else:
                with torch.no_grad():
                    # Pass save_timesteps parameter for early stopping evaluation
                    samples = self.model.sample(self.num_new_samples, save_timesteps=self.save_timesteps)
            
            samples_path = self.checkpoint_folder / f"fid_samples-epoch-{epoch+1}.pt"
            
            torch.save(samples, samples_path)

        self.save_checkpoint(epoch)

        # Calculate FID if enabled
        if self.calculate_fid:
            fid_score = self.fid_scorer.fid_score()
            self.accelerator.print(f'FID Score: {fid_score}')

            if self.save_best_and_latest_only:
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.save("best")
                self.save("latest")
