"""CLI entrypoint for online LoRA fine-tuning with arbitrage reward."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import config.config as config
from diffusion_factor_model import (
    ArbitrageValidator,
    ConditionalTransformer,
    OnlineDDPMLoRAFineTuner,
    SequentialGaussianDiffusion,
    make_arbitrage_reward_fn,
)


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
    diffusion = SequentialGaussianDiffusion(
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
    return diffusion


def infer_shape(data_path):
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


def get_git_metadata():
    repo_root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode()
        dirty = status.strip() != ""
    except Exception:
        commit = "unknown"
        status = ""
        dirty = False
    return {"commit": commit, "dirty": dirty, "status": status}


def save_checkpoint(path, diffusion, tuner, step, args, seq_len, state_shape):
    torch.save(
        {
            "model": diffusion.state_dict(),
            "optimizer": tuner.optimizer.state_dict(),
            "step": step,
            "state_shape": state_shape,
            "seq_len": seq_len,
            "hparams": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--gradient_mode", type=str, default="hybrid", choices=["direct", "reinforce", "hybrid"])
    parser.add_argument("--strict_grid_match", action="store_true", help="Require grid_size to match generated surface resolution")
    args = parser.parse_args()

    timestamp = int(time.time())
    run_name = f"ft_ts{timestamp}"
    results_root = Path(args.results_root) if args.results_root else (Path(config.MODELS_DIR) / "fine_tuning")
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.save_path is None:
        args.save_path = str(run_dir / "ft_lora_final.pt")

    git_meta = get_git_metadata()
    commit_label = git_meta["commit"] + (" (dirty)" if git_meta["dirty"] else "")
    print(f"Git commit: {commit_label}")
    print(f"Fine-tuning results folder: {run_dir}")

    with open(run_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(run_dir / "git_info.json", "w") as f:
        json.dump(git_meta, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len, state_shape = infer_shape(args.data_path)
    diffusion = build_model(seq_len, state_shape).to(device)

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    diffusion.load_state_dict(state, strict=False)

    m_vals = np.linspace(0.8, 1.2, args.grid_size)
    t_vals = np.linspace(0.1, 2.0, args.grid_size)
    validator = ArbitrageValidator(m_vals, t_vals, strict_grid_match=args.strict_grid_match)
    reward_fn = make_arbitrage_reward_fn(validator)

    tuner = OnlineDDPMLoRAFineTuner(
        diffusion,
        reward_fn=reward_fn,
        lr=args.lr,
        kl_weight=args.kl_weight,
        lora_rank=args.lora_rank,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    writer = SummaryWriter(log_dir=run_dir / "tb")
    checkpoint_paths = []

    pbar = tqdm(range(1, args.steps + 1), desc="Fine-tuning", unit="iter")
    for step in pbar:
        stats = tuner.step(args.batch_size)

        writer.add_scalar("FT/loss", stats.loss, step)
        writer.add_scalar("FT/policy_loss", stats.policy_loss, step)
        writer.add_scalar("FT/kl_loss", stats.kl_loss, step)
        writer.add_scalar("FT/grad_norm", stats.grad_norm, step)
        writer.add_scalar("FT/reward_mean", stats.reward_mean, step)
        writer.add_scalar("FT/reward_std", stats.reward_std, step)

        pbar.set_postfix(loss=f"{stats.loss:.4f}", kl=f"{stats.kl_loss:.4f}", grad=f"{stats.grad_norm:.4f}")

        if step % args.save_every == 0:
            ckpt_path = run_dir / f"model-step-{step}.pt"
            save_checkpoint(ckpt_path, diffusion, tuner, step, args, seq_len, state_shape)
            checkpoint_paths.append(str(ckpt_path))
            print(f"Saved fine-tuning checkpoint: {ckpt_path}")

    final_path = Path(args.save_path)
    if not final_path.is_absolute():
        final_path = run_dir / final_path
    save_checkpoint(final_path, diffusion, tuner, args.steps, args, seq_len, state_shape)

    writer.flush()
    writer.close()

    run_summary = {
        "final_checkpoint": str(final_path),
        "intermediate_checkpoints": checkpoint_paths,
        "results_dir": str(run_dir),
        "git_commit": commit_label,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Saved final fine-tuned checkpoint to {final_path}")
    print(f"Run summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
