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
    parser.add_argument(
        "--channel-mode",
        choices=["paper", "iv", "legacy", "custom"],
        default="paper",
        help="paper: log_iv, call_mid_over_s, log_return_broadcast; iv: log_iv, log_return_broadcast; legacy: alias for iv; custom: use --channels.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channels for --channel-mode custom. Supported: log_iv, call_mid_over_s, log_return_broadcast.",
    )
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


def parse_channels(channels_text: str) -> list[str]:
    channels = [item.strip() for item in channels_text.split(",") if item.strip()]
    supported = {"log_iv", "call_mid_over_s", "log_return_broadcast"}
    unknown = [channel for channel in channels if channel not in supported]
    if unknown:
        raise ValueError(f"unsupported channels: {unknown}; supported channels are {sorted(supported)}")
    if not channels:
        raise ValueError("at least one channel is required")
    return channels


def build_channel_array(tensor: np.lib.npyio.NpzFile, channel: str) -> np.ndarray:
    if channel == "log_return_broadcast":
        log_iv = np.asarray(tensor["log_iv"], dtype=np.float32)
        values = np.asarray(tensor["log_return"], dtype=np.float32)
        return np.broadcast_to(values[:, None, None], log_iv.shape).astype(np.float32)
    values = np.asarray(tensor[channel], dtype=np.float32)
    if channel == "call_mid_over_s" and np.any(values < 0):
        raise ValueError("call_mid_over_s contains negative values")
    return values


def channels_for_mode(args: argparse.Namespace) -> list[str]:
    if args.channel_mode == "paper":
        return ["log_iv", "call_mid_over_s", "log_return_broadcast"]
    if args.channel_mode in {"iv", "legacy"}:
        return ["log_iv", "log_return_broadcast"]
    if args.channels is None:
        raise ValueError("--channels is required when --channel-mode custom")
    return parse_channels(args.channels)


def channel_slug(channels: list[str]) -> str:
    if channels == ["log_iv", "call_mid_over_s", "log_return_broadcast"]:
        return "logiv_call_return"
    if channels == ["log_iv", "log_return_broadcast"]:
        return "logiv_return"
    parts = []
    for channel in channels:
        parts.append(
            channel.replace("_mid_over_s", "")
            .replace("_broadcast", "")
            .replace("log_iv", "logiv")
            .replace("log_return", "return")
        )
    return "_".join(parts)


def build_windows(tensor: np.lib.npyio.NpzFile, seq_len: int, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    channel_arrays = [build_channel_array(tensor, channel) for channel in channels]
    log_iv = np.asarray(tensor["log_iv"], dtype=np.float32)
    dates = np.asarray(tensor["dates"]).astype(str)
    if log_iv.ndim != 3:
        raise ValueError(f"log_iv must have shape [T, moneyness, tau], got {log_iv.shape}")
    if any(arr.shape != log_iv.shape for arr in channel_arrays):
        raise ValueError("all selected channels must have shape [T, moneyness, tau]")
    if log_iv.shape[0] != dates.shape[0]:
        raise ValueError("date and tensor lengths do not agree")
    if log_iv.shape[0] < seq_len:
        raise ValueError(f"need at least {seq_len} accepted dates, got {log_iv.shape[0]}")
    if any(not np.all(np.isfinite(arr)) for arr in channel_arrays):
        raise ValueError("non-finite selected channel in shared-grid tensor")

    num_windows = log_iv.shape[0] - seq_len + 1
    windows = np.empty((num_windows, seq_len, len(channels), log_iv.shape[2], log_iv.shape[1]), dtype=np.float32)
    window_dates = np.empty((num_windows, seq_len), dtype=object)
    for i in range(num_windows):
        for channel_index, arr in enumerate(channel_arrays):
            windows[i, :, channel_index] = np.transpose(arr[i : i + seq_len], (0, 2, 1))
        window_dates[i] = dates[i : i + seq_len]
    return windows, window_dates.astype(str)


def write_outputs(output_dir: Path, grid: dict, windows: np.ndarray, window_dates: np.ndarray, args: argparse.Namespace) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.seq_len}d"
    data_path = output_dir / f"shared_grid_{slug}_{channel_slug(args.channels_list)}.npy"
    conditioning_path = output_dir / f"shared_grid_{slug}_conditioning.npy"
    dates_path = output_dir / f"shared_grid_{slug}_dates.npy"
    metadata_path = output_dir / f"shared_grid_{slug}_metadata.json"
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
        "channels": args.channels_list,
        "channel_mode": args.channel_mode,
        "conditioning_target": "next_trading_day_after_observed_prefix",
        "seq_len": args.seq_len,
        "conditioning_length": args.conditioning_length,
        "target_index": args.conditioning_length,
        "moneyness_grid": grid["moneyness_grid"],
        "tau_grid": grid["tau_grid"],
        "rolling_windows": True,
        "window_stride": 1,
        "dtype": str(windows.dtype),
        "min_max": {
            channel: [float(windows[:, :, i].min()), float(windows[:, :, i].max())]
            for i, channel in enumerate(args.channels_list)
        },
        "recommended_train_command": (
            "python train.py --data_path {data} --conditioning_path {cond} "
            "--conditioning_length {clen} --window_length {seq}"
        ).format(data=data_path, cond=conditioning_path, clen=args.conditioning_length, seq=args.seq_len),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def self_check(output_dir: Path, expected_seq_len: int, expected_conditioning_length: int, channels: list[str], seq_len: int) -> None:
    slug = f"{seq_len}d"
    data = np.load(output_dir / f"shared_grid_{slug}_{channel_slug(channels)}.npy")
    cond = np.load(output_dir / f"shared_grid_{slug}_conditioning.npy")
    dates = np.load(output_dir / f"shared_grid_{slug}_dates.npy", allow_pickle=True)
    metadata = json.loads((output_dir / f"shared_grid_{slug}_metadata.json").read_text())
    assert data.shape == cond.shape
    assert data.ndim == 5
    assert data.shape[1] == expected_seq_len
    assert data.shape[2] == len(metadata["channels"])
    assert data.shape[2] == len(channels)
    assert data.shape[3:] == (9, 11)
    assert dates.shape[:2] == data.shape[:2]
    assert np.all(np.isfinite(data))
    assert np.allclose(data, cond)
    assert metadata["layout"] == "N,S,C,H,W"
    assert metadata["axis_convention"] == {"H": "tau/maturity", "W": "moneyness"}
    assert metadata["conditioning_length"] == expected_conditioning_length
    assert metadata["channels"] == channels
    assert len(metadata["moneyness_grid"]) == 11
    assert len(metadata["tau_grid"]) == 9
    if "call_mid_over_s" in channels:
        call_index = channels.index("call_mid_over_s")
        assert np.all(data[:, :, call_index] >= 0)


def main() -> None:
    args = parse_args()
    if args.conditioning_length < 0 or args.conditioning_length >= args.seq_len:
        raise ValueError("--conditioning-length must be between 0 and seq_len - 1")
    args.channels_list = channels_for_mode(args)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    grid, tensor = load_shared_grid(processed_dir)
    windows, dates = build_windows(tensor, args.seq_len, args.channels_list)
    metadata = write_outputs(output_dir, grid, windows, dates, args)
    if args.self_check:
        self_check(output_dir, args.seq_len, args.conditioning_length, args.channels_list, seq_len=args.seq_len)
        print("SHARED_GRID_DIFFUSION_DATA_SELF_CHECK=PASS")
    print(json.dumps({"data_path": metadata["data_path"], "conditioning_path": metadata["conditioning_path"], "shape": metadata["shape"]}, sort_keys=True))


if __name__ == "__main__":
    main()
