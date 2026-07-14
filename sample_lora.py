"""Sample from a LoRA fine-tuned diffusion checkpoint without re-training."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import config.config as config
from adapted_sequential_diffusion import ConditionalTransformer, SequentialGaussianDiffusion
from adapted_sequential_diffusion.fine_tuning import inject_lora


def infer_shape(data_path: str):
    data = np.load(data_path)
    if data.ndim == 5:
        return int(data.shape[1]), tuple(data.shape[2:])
    if data.ndim == 4:
        return int(data.shape[1]), (1, int(data.shape[2]), int(data.shape[3]))
    if data.ndim == 3:
        return int(data.shape[1]), (int(data.shape[2]),)
    if data.ndim == 2:
        return int(data.shape[1]), ()
    raise ValueError(f"Unsupported data shape: {data.shape}")


def build_model(seq_len, state_shape):
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
    return SequentialGaussianDiffusion(
        model,
        seq_len=seq_len,
        timesteps=config.TIMESTEPS,
        sampling_timesteps=config.SAMPLING_TIMESTEPS,
        ddim_eta=config.DDIM_ETA,
        objective=config.OBJECTIVE,
        beta_schedule=config.BETA_SCHEDULE,
        auto_normalize=config.AUTO_NORMALIZE,
        state_shape=state_shape,
    )


def parse_target_modules(raw: str):
    return tuple([item.strip() for item in raw.split(",") if item.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned LoRA checkpoint produced by fine_tune.py")
    parser.add_argument("--data_path", type=str, required=True, help="Training data path used for normalization stats")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path")
    parser.add_argument("--sample_batches", type=int, default=1)
    parser.add_argument("--samples_per_batch", type=int, default=64)
    parser.add_argument("--save_timesteps", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="encoder,output,value_proj,time_mlp",
        help="Comma-separated module name substrings for LoRA injection",
    )
    args = parser.parse_args()

    if args.seed is not None:
        config.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len, state_shape = infer_shape(args.data_path)

    data_np = np.load(args.data_path)
    if data_np.ndim == 4:
        data = torch.from_numpy(data_np).float().unsqueeze(2)
    else:
        data = torch.from_numpy(data_np).float()
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)

    diffusion = build_model(seq_len, state_shape).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    ckpt_rank = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("hparams"), dict):
        ckpt_rank = ckpt["hparams"].get("lora_rank")
    lora_rank = int(args.lora_rank if args.lora_rank is not None else (ckpt_rank if ckpt_rank is not None else 8))

    inject_lora(
        diffusion.model,
        rank=lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.lora_target_modules),
    )

    missing, unexpected = diffusion.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {unexpected[:20]}")
    if missing:
        print(f"Warning: missing keys during load (showing up to 20): {missing[:20]}")

    diffusion.eval()

    sampled_batches = []
    for i in tqdm(range(args.sample_batches), desc="Sampling", unit="batch"):
        try:
            samples = diffusion.sample(
                batch_size=args.samples_per_batch,
                save_timesteps=args.save_timesteps,
                show_progress=True,
                progress_desc=f"Sampling batch {i + 1}/{args.sample_batches}",
            )
        except TypeError:
            samples = diffusion.sample(
                batch_size=args.samples_per_batch,
                save_timesteps=args.save_timesteps,
            )
        samples = samples.cpu()
        samples = samples * data_std.to(samples.device) + data_mean.to(samples.device)
        sampled_batches.append(samples.numpy())

    all_samples = np.concatenate(sampled_batches, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_samples)
    print(f"Saved samples to {output_path} with shape {all_samples.shape}")


if __name__ == "__main__":
    main()
