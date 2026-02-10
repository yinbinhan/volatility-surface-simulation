"""
Training script for Diffusion Factor Model
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import os
import gc
import argparse
import time
import json
import subprocess
from tqdm import tqdm


from diffusion_factor_model.diffusion_factor_model import (
    ConditionalTransformer,
    SequentialGaussianDiffusion,
    Trainer,
)
import config.config as config

def get_dim_mults_for_size(height, width):
    """
    Determine appropriate dimension multipliers for UNet based on input dimensions.
    The dimensions must be divisible by the maximum downsampling factor.
    
    Args:
        height: Height of input
        width: Width of input
        
    Returns:
        Tuple of dimension multipliers suitable for the input size
    """
    # Calculate the maximum downsampling factor possible
    min_dim = min(height, width)
    
    if min_dim >= 32:
        return config.DIM_MULTS_LARGE  # Standard for large inputs
    elif min_dim >= 16:
        return config.DIM_MULTS_MEDIUM  # For medium inputs
    elif min_dim >= 8:
        return config.DIM_MULTS_SMALL   # For small inputs
    elif min_dim >= 4:
        return config.DIM_MULTS_TINY    # For very small inputs
    else:
        return config.DIM_MULTS_MINIMAL # Minimal case

def train_model(data_path, seed=None, num_samples=None, gpu_id=0, epochs=None, save_timesteps=None, num_features=None, window_start=0, window_length=None, conditioning_path=None, conditioning_length=None, checkpoint_path=None, skip_training=False, cli_args=None):
    """
    Train the diffusion model using a specific data file
    
    Args:
        data_path: Path to the data file to use for training
        seed: Random seed for reproducibility
        num_samples: Number of training samples to use (None = use all)
        gpu_id: GPU ID to use
        epochs: Number of epochs to train (None = use config.EPOCHS)
        save_timesteps: List of specific timesteps to save during sampling for early stopping evaluation
                       (None = use config.SAVE_TIMESTEPS, which defaults to None meaning save only final result)
        sample_window_start: Optional start index (inclusive) for sequential sampling
        sample_window_length: Optional number of sequential entries to generate
        conditioning_path: Optional path to a conditioning sequence file for sampling
        conditioning_length: Optional number of prefix entries to condition on during sampling
        checkpoint_path: Optional path to a saved checkpoint to load before training/sampling
        skip_training: If True, load the checkpoint (if provided) and skip training to only run sampling
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if cli_args is None:
        cli_args = {}

    def get_git_metadata():
        """Return commit hash plus dirty state and optional status/diff snapshots."""

        repo_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            repo_root = (
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=repo_dir,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=repo_root,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
            )
            dirty = status.strip() != ""
            try:
                diff = (
                    subprocess.check_output(
                        ["git", "diff"], cwd=repo_root, stderr=subprocess.DEVNULL
                    )
                    .decode()
                )
            except Exception:
                diff = None
            return {
                "commit": commit,
                "dirty": dirty,
                "status": status,
                "diff": diff,
            }
        except Exception:
            return {
                "commit": "unknown",
                "dirty": False,
                "status": None,
                "diff": None,
            }
    
    # Use config default if save_timesteps not specified
    if save_timesteps is None:
        save_timesteps = config.SAVE_TIMESTEPS
    
    # Set seed and get timestamp for experiment ID
    seed = config.set_seed(seed)
    timestamp = int(time.time())
    
    # Get filename from path for experiment ID
    filename = os.path.basename(data_path)
    data_id = os.path.splitext(filename)[0]
    
    # Create experiment ID
    exp_id = f"{config.EXP_PREFIX}_{data_id}_ts{timestamp}_seed{seed}"

    git_info = get_git_metadata()
    commit_hash = git_info["commit"]
    commit_label = commit_hash + (" (dirty)" if git_info["dirty"] else "")
    print(f"Git commit hash: {commit_label}")
    if git_info["dirty"]:
        print("Working tree has uncommitted changes; storing status and diff with the run.")
    
    # Load data to determine shape and dimensions
    data_np = np.load(data_path)
    data_shape = data_np.shape
    print(f"Loaded data with shape: {data_shape}, dtype: {data_np.dtype}")
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < data_shape[0]:
        data_np = data_np[:num_samples]
        print(f"Using {num_samples} samples from the data")

    # Limit number of features if specified
    if num_features is not None and num_features < data_shape[1]:
        data_np = data_np[:, :num_features]
        print(f"Using {num_features} features from the data")
        data_shape = data_np.shape
    
    # Determine data dimensions and reshape strategy for sequential diffusion
    # Expected canonical layout is [samples, seq_len, channel, height, width], but we keep
    # backward compatibility with scalar/vector-per-step layouts.
    if len(data_shape) == 2:
        # [samples, seq_len] -> scalar state per step
        data = torch.from_numpy(data_np).float()
    elif len(data_shape) == 3:
        # [samples, seq_len, features] -> vector state per step
        data = torch.from_numpy(data_np).float()
    elif len(data_shape) == 4:
        # [samples, seq_len, height, width] -> add channel dim
        data = torch.from_numpy(data_np).float().unsqueeze(2)
    elif len(data_shape) == 5:
        # [samples, seq_len, channel, height, width]
        data = torch.from_numpy(data_np).float()
    else:
        raise ValueError(
            f"Unsupported data shape {data_shape}. Expected [N,S], [N,S,F], [N,S,H,W], or [N,S,C,H,W]."
        )

    total_seq_len = data.shape[1]
    if window_length is None:
        window_end = total_seq_len
    else:
        window_end = min(total_seq_len, int(window_start) + max(1, int(window_length)))

    if window_start >= total_seq_len:
        raise ValueError(
            f"Sampling window start {window_start} exceeds sequence length {total_seq_len}"
        )
    if window_end - window_start <= 0:
        raise ValueError("Sampling window must include at least one index")

    if window_start != 0 or window_end != total_seq_len:
        print(
            f"Restricting training data to indices [{window_start}, {window_end}) out of {total_seq_len}"
        )

    data = data[:, window_start:window_end]
    samples, seq_len = data.shape[:2]
    state_shape = tuple(data.shape[2:])
    print(f"Using sequence data with length {seq_len}, state shape {state_shape}, and {samples} samples")

    # Create directories for this experiment
    model_dir = os.path.join(config.MODELS_DIR, exp_id)
    sample_dir = os.path.join(config.SAMPLES_DIR, exp_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Save commit hash for reproducibility
    hash_record = os.path.join(model_dir, "commit_hash.txt")
    try:
        with open(hash_record, "w") as f:
            f.write(commit_label + "\n")
    except OSError:
        print(f"Warning: unable to write commit hash to {hash_record}")

    if git_info["dirty"]:
        status_record = os.path.join(model_dir, "git_status.txt")
        diff_record = os.path.join(model_dir, "git_diff.patch")
        try:
            with open(status_record, "w") as f:
                f.write(git_info["status"])
            print(f"Saved git status to {status_record}")
        except OSError:
            print(f"Warning: unable to write git status to {status_record}")
        if git_info["diff"] is not None:
            try:
                with open(diff_record, "w") as f:
                    f.write(git_info["diff"])
                print(f"Saved git diff to {diff_record}")
            except OSError:
                print(f"Warning: unable to write git diff to {diff_record}")

    # Persist CLI arguments and config snapshot for reproducibility
    run_record = os.path.join(model_dir, "run_config.json")
    config_snapshot = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    metadata = {
        "commit_hash": commit_hash,
        "commit_dirty": git_info["dirty"],
        "cli_args": cli_args if cli_args is not None else {},
        "config": config_snapshot,
    }
    try:
        with open(run_record, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved run configuration to {run_record}")
    except OSError:
        print(f"Warning: unable to write run configuration to {run_record}")

    # Create dataset
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
    normalized_data = (data - data_mean) / data_std
    dataset = TensorDataset(normalized_data)

    conditioning_source = None
    if conditioning_path is not None:
        conditioning_np = np.load(conditioning_path)
        if conditioning_np.shape[1] != total_seq_len:
            raise ValueError(
                f"Conditioning sequence length {conditioning_np.shape[1]} "
                f"does not match data length {total_seq_len}"
            )
        if tuple(conditioning_np.shape[2:]) != state_shape:
            raise ValueError(
                f"Conditioning state shape {tuple(conditioning_np.shape[2:])} does not match data state shape {state_shape}"
            )
        conditioning = torch.from_numpy(conditioning_np).float()
        conditioning = conditioning[:, window_start:window_end]
        conditioning_source = (conditioning - data_mean) / data_std

    if conditioning_length is None:
        conditioning_length = seq_len if conditioning_source is not None else 0
    conditioning_length = int(conditioning_length)
    if conditioning_length < 0 or conditioning_length > seq_len:
        raise ValueError("conditioning_length must be between 0 and the sequence length")

    # Use epochs from argument or config
    if epochs is None:
        epochs = config.EPOCHS
    
    # Initialize sequential transformer model
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

    print("Model initialized")

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
    
    print("Diffusion process initialized")

    # Initialize Trainer with custom epochs and optional save_timesteps for early stopping
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=min(config.BATCH_SIZE, len(dataset)),  # Ensure batch size doesn't exceed dataset size
        train_lr=config.LEARNING_RATE,
        train_epochs=epochs,
        adamw_weight_decay=config.WEIGHT_DECAY,
        cosine_scheduler=config.USE_COSINE_SCHEDULER,
        warm_up=config.USE_WARM_UP,
        warmup_iters=config.WARMUP_STEPS,
        T_0=config.COSINE_CYCLE_LENGTH,
        T_mult=config.T_MULT,
        eta_min=config.COSINE_LR_MIN,
        cosine_steps=config.COSINE_STEPS,
        gradient_accumulate_every=config.GRADIENT_ACCUMULATION,
        ema_decay=config.EMA_DECAY,
        split_batches=config.SPLIT_BATCHES,
        save_and_sample_every=config.SAVE_INTERVAL,
        results_folder=model_dir,
        param_path="",
        amp=config.USE_AMP,
        save_timesteps=save_timesteps,  # Pass save_timesteps for early stopping evaluation
    )

    print("Trainer initialized")
    print(f"Models saved to: {model_dir}")

    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        print(f"Loading checkpoint from {checkpoint_path}")
        trainer.load(checkpoint_path)
        print("Checkpoint loaded")
    elif skip_training:
        raise ValueError("--skip_training requires a valid --checkpoint_path to load weights")

    # Train model unless explicitly skipped
    if skip_training:
        print("Skipping training and proceeding directly to sampling")
        diffusion.eval()
    else:
        print(f"Starting training for {epochs} epochs...")
        trainer.train()
        diffusion.eval()
    
    # Generate samples
    print("Generating samples...")
    print(f"Samples saved to: {sample_dir}")
    sample_batches = config.SAMPLE_BATCHES
    samples_per_batch = config.SAMPLES_PER_BATCH
    
    config.set_seed(seed)  # Reset seed for reproducibility
    
    for i in range(sample_batches):
        # Pass save_timesteps parameter to sample method for early stopping evaluation
        try:
            samples = diffusion.sample(
                batch_size=samples_per_batch,
                save_timesteps=save_timesteps,
                show_progress=True,
                progress_desc=f"Sampling batch {i+1}/{sample_batches}",
            )
        except TypeError:
            samples = diffusion.sample(batch_size=samples_per_batch, save_timesteps=save_timesteps)
        samples = samples.view(samples.size(0), -1).cpu().numpy()
        samples = samples * data_std.view(-1).cpu().numpy() + data_mean.view(-1).cpu().numpy()
        
        sample_file = os.path.join(sample_dir, f"sample_batch{i+1}.npy")
        np.save(sample_file, samples)
        
        # Clean up to prevent memory issues
        del samples
        gc.collect()
    
    # Clean up
    del trainer, model, diffusion, data, dataset
    gc.collect()
    
    print(f"Training and sampling complete for {exp_id}")
    print(f"Models saved to: {model_dir}")
    print(f"Samples saved to: {sample_dir}")
    
    return model_dir, sample_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion factor model on specific data file")
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to the data file for training")
    parser.add_argument("--seed", type=int, default=None, 
                      help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None, 
                      help="Number of training samples (None = use all)")
    parser.add_argument("--num_features", type=int, default=None, 
                      help="Number of features (None = use all)")
    parser.add_argument("--gpu", type=int, default=0, 
                      help="GPU ID")
    parser.add_argument("--epochs", type=int, default=None, 
                      help="Number of epochs to train (None = use config value)")
    parser.add_argument("--save_timesteps", type=int, nargs='+', default=None,
                      help="Specific timesteps to save during sampling for early stopping evaluation (e.g., --save_timesteps 100 200 500)")
    parser.add_argument("--window_start", type=int, default=0,
                      help="Sequence window start index")
    parser.add_argument("--window_length", type=int, default=None,
                      help="Sequence window length")
    parser.add_argument("--conditioning_path", type=str, default=None,
                      help="Optional conditioning tensor path with shape [N,S,*state_shape]")
    parser.add_argument("--conditioning_length", type=int, default=None,
                      help="Number of sequence steps to condition during sampling")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                      help="Checkpoint to load before training/sampling")
    parser.add_argument("--skip_training", action="store_true",
                      help="Skip training and only run sampling (requires --checkpoint_path)")
    
    args = parser.parse_args()
    
    train_model(args.data_path, args.seed, args.num_samples, args.gpu, args.epochs, args.save_timesteps, num_features=args.num_features, window_start=args.window_start, window_length=args.window_length, conditioning_path=args.conditioning_path, conditioning_length=args.conditioning_length, checkpoint_path=args.checkpoint_path, skip_training=args.skip_training, cli_args=vars(args)) 
