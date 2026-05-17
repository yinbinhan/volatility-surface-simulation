#!/usr/bin/env python3
"""Prepare rolling diffusion windows from shared-grid VolGAN artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True, help="Directory produced by VolGAN shared_grid_preprocessing.py")
    parser.add_argument("--output-dir", default="data/shared_grid_11x9")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--conditioning-length", type=int, default=29)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def load_shared_grid(processed_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    grid_path = processed_dir / "grid_config.json"
    tensor_path = processed_dir / "surface_tensor.npz"
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    grid = json.loads(grid_path.read_text())
    tensor = np.load(tensor_path)
    if grid.get("grid_order") != "m_major_tau_minor":
        raise ValueError("expected VolGAN shared grid order m_major_tau_minor")
    return grid, tensor


def build_windows(tensor: np.lib.npyio.NpzFile, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    log_iv = np.asarray(tensor["log_iv"], dtype=np.float32)
    log_return = np.asarray(tensor["log_return"], dtype=np.float32)
    dates = np.asarray(tensor["dates"]).astype(str)
    if log_iv.ndim != 3:
        raise ValueError(f"log_iv must have shape [T, moneyness, tau], got {log_iv.shape}")
    if log_iv.shape[0] != log_return.shape[0] or log_iv.shape[0] != dates.shape[0]:
        raise ValueError("date, log_iv, and log_return lengths do not agree")
    if log_iv.shape[0] < seq_len:
        raise ValueError(f"need at least {seq_len} accepted dates, got {log_iv.shape[0]}")
    if not np.all(np.isfinite(log_iv)) or not np.all(np.isfinite(log_return)):
        raise ValueError("non-finite log_iv or log_return in shared-grid tensor")

    num_windows = log_iv.shape[0] - seq_len + 1
    windows = np.empty((num_windows, seq_len, 2, log_iv.shape[2], log_iv.shape[1]), dtype=np.float32)
    window_dates = np.empty((num_windows, seq_len), dtype=object)
    for i in range(num_windows):
        iv_path = log_iv[i : i + seq_len]
        rtn_path = log_return[i : i + seq_len]
        windows[i, :, 0] = np.transpose(iv_path, (0, 2, 1))
        windows[i, :, 1] = rtn_path[:, None, None]
        window_dates[i] = dates[i : i + seq_len]
    return windows, window_dates.astype(str)


def write_outputs(output_dir: Path, grid: dict, windows: np.ndarray, window_dates: np.ndarray, args: argparse.Namespace) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "shared_grid_30d_logiv_return.npy"
    conditioning_path = output_dir / "shared_grid_30d_conditioning.npy"
    dates_path = output_dir / "shared_grid_30d_dates.npy"
    metadata_path = output_dir / "shared_grid_30d_metadata.json"
    np.save(data_path, windows)
    np.save(conditioning_path, windows)
    np.save(dates_path, window_dates)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_processed_dir": str(Path(args.processed_dir)),
        "data_path": str(data_path),
        "conditioning_path": str(conditioning_path),
        "dates_path": str(dates_path),
        "shape": list(windows.shape),
        "layout": "N,S,C,H,W",
        "axis_convention": {"H": "tau/maturity", "W": "moneyness"},
        "channels": ["log_iv", "log_return_broadcast"],
        "seq_len": args.seq_len,
        "conditioning_length": args.conditioning_length,
        "target_index": args.conditioning_length,
        "moneyness_grid": grid["moneyness_grid"],
        "tau_grid": grid["tau_grid"],
        "rolling_windows": True,
        "window_stride": 1,
        "dtype": str(windows.dtype),
        "min_max": {
            "log_iv": [float(windows[:, :, 0].min()), float(windows[:, :, 0].max())],
            "log_return": [float(windows[:, :, 1].min()), float(windows[:, :, 1].max())],
        },
        "recommended_train_command": (
            "python train.py --data_path {data} --conditioning_path {cond} "
            "--conditioning_length {clen} --window_length {seq}"
        ).format(data=data_path, cond=conditioning_path, clen=args.conditioning_length, seq=args.seq_len),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def self_check(output_dir: Path, expected_seq_len: int, expected_conditioning_length: int) -> None:
    data = np.load(output_dir / "shared_grid_30d_logiv_return.npy")
    cond = np.load(output_dir / "shared_grid_30d_conditioning.npy")
    dates = np.load(output_dir / "shared_grid_30d_dates.npy", allow_pickle=True)
    metadata = json.loads((output_dir / "shared_grid_30d_metadata.json").read_text())
    assert data.shape == cond.shape
    assert data.ndim == 5
    assert data.shape[1:] == (expected_seq_len, 2, 9, 11)
    assert dates.shape[:2] == data.shape[:2]
    assert np.all(np.isfinite(data))
    assert np.allclose(data, cond)
    assert metadata["layout"] == "N,S,C,H,W"
    assert metadata["axis_convention"] == {"H": "tau/maturity", "W": "moneyness"}
    assert metadata["conditioning_length"] == expected_conditioning_length
    assert metadata["channels"] == ["log_iv", "log_return_broadcast"]
    assert len(metadata["moneyness_grid"]) == 11
    assert len(metadata["tau_grid"]) == 9


def main() -> None:
    args = parse_args()
    if args.conditioning_length < 0 or args.conditioning_length >= args.seq_len:
        raise ValueError("--conditioning-length must be between 0 and seq_len - 1")
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    grid, tensor = load_shared_grid(processed_dir)
    windows, dates = build_windows(tensor, args.seq_len)
    metadata = write_outputs(output_dir, grid, windows, dates, args)
    if args.self_check:
        self_check(output_dir, args.seq_len, args.conditioning_length)
        print("SHARED_GRID_DIFFUSION_DATA_SELF_CHECK=PASS")
    print(json.dumps({"data_path": metadata["data_path"], "conditioning_path": metadata["conditioning_path"], "shape": metadata["shape"]}, sort_keys=True))


if __name__ == "__main__":
    main()
