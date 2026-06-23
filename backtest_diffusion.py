"""Diffusion hedging backtest using 21-day conditioned one-step scenarios."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config.config as config
from diffusion_factor_model import ConditionalTransformer, SequentialGaussianDiffusion
from backtest_volgan import (
    RISK_FREE,
    assemble_total_greeks,
    assemble_total_scenarios,
    assemble_total_vector,
    bs_price_from_surface,
    build_instrument_panel,
    build_state_lookup,
    get_day_state,
    get_half_spreads,
    load_delta_surface,
    print_table2,
    scenarios_to_solver_arrays,
    select_alpha_aic,
    solve_transaction_cost_lasso,
    split_hedge_universe,
    tracking_error_stats,
    _hedging,
    _ds_greeks,
    _ds_price,
    _set_tau,
)

BENCHMARK_VEGA_FLOOR = _hedging.BENCHMARK_VEGA_FLOOR


@dataclass
class DiffusionContext:
    diffusion: SequentialGaussianDiffusion
    data_mean: torch.Tensor
    data_std: torch.Tensor
    log_iv: np.ndarray
    log_return: np.ndarray
    dates: list[pd.Timestamp]
    date_to_idx: dict[pd.Timestamp, int]
    m_grid: np.ndarray
    tau_grid: np.ndarray
    device: torch.device


def _to_training_tensor(data_np: np.ndarray) -> torch.Tensor:
    if data_np.ndim == 4:
        return torch.from_numpy(data_np).float().unsqueeze(2)
    return torch.from_numpy(data_np).float()


def _infer_shape(data_path: Path) -> tuple[int, tuple[int, ...]]:
    data = np.load(data_path, mmap_mode="r")
    if data.ndim == 5:
        return int(data.shape[1]), tuple(int(x) for x in data.shape[2:])
    if data.ndim == 4:
        return int(data.shape[1]), (1, int(data.shape[2]), int(data.shape[3]))
    if data.ndim == 3:
        return int(data.shape[1]), (int(data.shape[2]),)
    if data.ndim == 2:
        return int(data.shape[1]), ()
    raise ValueError(f"unsupported data shape {data.shape}")


def _build_model(seq_len: int, state_shape: tuple[int, ...]) -> SequentialGaussianDiffusion:
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


def load_diffusion_context(
    checkpoint: Path,
    train_data_path: Path,
    processed_dir: Path,
    device_name: str,
) -> DiffusionContext:
    device = torch.device(device_name)
    seq_len, state_shape = _infer_shape(train_data_path)
    if seq_len != 22 or state_shape != (2, 9, 11):
        raise ValueError(f"expected 22-step IV/return data with state_shape (2,9,11), got {seq_len}, {state_shape}")

    train_np = np.load(train_data_path)
    train_tensor = _to_training_tensor(train_np)
    data_mean = train_tensor.mean(dim=0, keepdim=True)
    data_std = train_tensor.std(dim=0, keepdim=True)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)

    diffusion = _build_model(seq_len, state_shape).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    diffusion.load_state_dict(state, strict=True)
    diffusion.eval()

    tensor = np.load(processed_dir / "surface_tensor.npz")
    grid = json.loads((processed_dir / "grid_config.json").read_text())
    log_iv = np.asarray(tensor["log_iv"], dtype=np.float32)
    log_return = np.asarray(tensor["log_return"], dtype=np.float32)
    dates = [pd.Timestamp(str(d)) for d in np.asarray(tensor["dates"]).astype(str)]
    if log_iv.shape[1:] != (11, 9):
        raise ValueError(f"expected processed log_iv shape [T,11,9], got {log_iv.shape}")
    return DiffusionContext(
        diffusion=diffusion,
        data_mean=data_mean,
        data_std=data_std,
        log_iv=log_iv,
        log_return=log_return,
        dates=dates,
        date_to_idx={date: i for i, date in enumerate(dates)},
        m_grid=np.asarray(grid["moneyness_grid"], dtype=float),
        tau_grid=np.asarray(grid["tau_grid"], dtype=float),
        device=device,
    )


def sample_diffusion_scenarios(
    ctx: DiffusionContext,
    date_t: pd.Timestamp,
    spot_t: float,
    n_scenarios: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    idx = ctx.date_to_idx.get(pd.Timestamp(date_t))
    if idx is None or idx < 20:
        return None

    prefix_iv = ctx.log_iv[idx - 20 : idx + 1]  # [21, m, tau]
    prefix_ret = ctx.log_return[idx - 20 : idx + 1]
    prefix = np.empty((21, 2, 9, 11), dtype=np.float32)
    prefix[:, 0] = np.transpose(prefix_iv, (0, 2, 1))
    prefix[:, 1] = np.broadcast_to(prefix_ret[:, None, None], (21, 9, 11))

    spots = []
    ivs = []
    mask = torch.zeros((22,), dtype=torch.bool, device=ctx.device)
    mask[:21] = True
    generated = 0
    while generated < n_scenarios:
        current = min(batch_size, n_scenarios - generated)
        cond = np.zeros((current, 22, 2, 9, 11), dtype=np.float32)
        cond[:, :21] = prefix[None, :]
        cond_t = torch.from_numpy(cond).float()
        cond_t = (cond_t - ctx.data_mean) / ctx.data_std
        cond_t = cond_t.to(ctx.device)
        with torch.no_grad():
            sample = ctx.diffusion.sample(
                batch_size=current,
                conditioning=cond_t,
                conditioning_mask=mask,
                start_idx=21,
                end_idx=22,
                show_progress=False,
            )
        sample = sample.cpu() * ctx.data_std + ctx.data_mean
        target = sample.numpy()[:, 21]
        log_ret_next = target[:, 1].mean(axis=(1, 2))
        spots.append(float(spot_t) * np.exp(log_ret_next))
        ivs.append(np.exp(target[:, 0].transpose(0, 2, 1)))  # [N, m, tau]
        generated += current
    return np.concatenate(spots), np.concatenate(ivs, axis=0)


def run_one_window(
    panel,
    ctx: DiffusionContext,
    state_lookup,
    n_scenarios: int,
    n_val: int,
    batch_size: int,
    delta_surface_lookup: dict[pd.Timestamp, pd.DataFrame],
):
    dates, log_iv_rows, closes, log_rets, date_to_idx, m_grid, tau_grid, grid_order = state_lookup
    trading_dates = panel.trading_dates
    sorted_hedges, option_hedges, option_indices, underlying_idx = split_hedge_universe(panel)
    hedge_ids = list(sorted_hedges["optionid"])
    if panel.fixed_atm_optionid is None:
        return None
    atm_matches = sorted_hedges["optionid"].map(lambda value: str(value) == str(panel.fixed_atm_optionid)).to_numpy(dtype=bool)
    if int(atm_matches.sum()) != 1:
        return None
    atm_idx = int(np.flatnonzero(atm_matches)[0])

    target_contracts = panel.target.sort_values(["cp_flag", "strike"])[["cp_flag", "strike", "ttm"]].rename(columns={"ttm": "tau"}).reset_index(drop=True)
    hedge_contracts = option_hedges[["cp_flag", "strike", "ttm"]].rename(columns={"ttm": "tau"}).reset_index(drop=True)

    expiry = panel.expiry_date
    if len(trading_dates) < 2:
        return None

    t0_date = trading_dates[0]
    t0_state = get_day_state(t0_date, date_to_idx, log_iv_rows, closes, log_rets)
    if t0_state is None:
        return None
    log_iv_t0, spot_t0, *_ = t0_state
    tau_t0 = max((expiry - t0_date).days / 365, 1.0 / 365)
    tc_t0 = _set_tau(target_contracts, tau_t0)
    hc_t0 = _set_tau(hedge_contracts, tau_t0)

    t0_target_prices = bs_price_from_surface(log_iv_t0, spot_t0, tc_t0, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order)
    t0_hedge_prices_option = bs_price_from_surface(log_iv_t0, spot_t0, hc_t0, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order)
    V0 = float(t0_target_prices.sum())

    day0_df = delta_surface_lookup.get(pd.Timestamp(t0_date))
    if day0_df is None:
        return None
    V0_ds = float(_ds_price(day0_df, spot_t0, tc_t0, r=RISK_FREE).sum())
    if V0_ds <= 0:
        return None

    phi_zero = np.zeros(len(hedge_ids))
    c_t0 = get_half_spreads(panel.quotes, t0_date, hedge_ids)

    scenarios = sample_diffusion_scenarios(ctx, t0_date, spot_t0, n_scenarios, batch_size)
    validation = sample_diffusion_scenarios(ctx, t0_date, spot_t0, n_val, batch_size)
    if scenarios is None or validation is None:
        return None
    spots_tr, iv_tr = scenarios
    spots_val, iv_val = validation
    dV_tr, dH_tr_option = scenarios_to_solver_arrays(spots_tr, iv_tr, spot_t0, tc_t0, hc_t0, t0_target_prices, t0_hedge_prices_option, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid)
    dH_tr = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_tr_option, underlying_idx, spots_tr, spot_t0)
    dV_val, dH_val_option = scenarios_to_solver_arrays(spots_val, iv_val, spot_t0, tc_t0, hc_t0, t0_target_prices, t0_hedge_prices_option, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid)
    dH_val = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_val_option, underlying_idx, spots_val, spot_t0)
    alpha_best = select_alpha_aic(dV_tr, dH_tr, dV_val, dH_val, phi_prev=phi_zero, c_i=c_t0, g0_scale=V0)

    phi_diffusion = phi_zero.copy()
    Pi_diffusion = V0_ds
    Pi_delta = V0_ds
    Pi_dv = V0_ds
    phi_vega_atm = 0.0
    Z_diffusion, Z_delta, Z_dv, Z_unhedged = [], [], [], []

    for step in range(len(trading_dates) - 1):
        date_t = trading_dates[step]
        date_tp1 = trading_dates[step + 1]
        state_t = get_day_state(date_t, date_to_idx, log_iv_rows, closes, log_rets)
        state_tp1 = get_day_state(date_tp1, date_to_idx, log_iv_rows, closes, log_rets)
        if state_t is None or state_tp1 is None:
            break
        log_iv_t, spot_t, *_ = state_t
        _, spot_tp1, *_ = state_tp1
        tau_t = max((expiry - date_t).days / 365, 1.0 / 365)
        tau_tp1 = max((expiry - date_tp1).days / 365, 1.0 / 365)
        day_df_t = delta_surface_lookup.get(pd.Timestamp(date_t))
        day_df_tp1 = delta_surface_lookup.get(pd.Timestamp(date_tp1))
        if day_df_t is None or day_df_tp1 is None:
            break

        tc_ds_t = _set_tau(target_contracts, tau_t)
        hc_ds_t = _set_tau(hedge_contracts, tau_t)
        tc_ds_tp1 = _set_tau(target_contracts, tau_tp1)
        hc_ds_tp1 = _set_tau(hedge_contracts, tau_tp1)
        prices_target_t = _ds_price(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        prices_hedge_t_option = _ds_price(day_df_t, spot_t, hc_ds_t, r=RISK_FREE)
        prices_target_tp1 = _ds_price(day_df_tp1, spot_tp1, tc_ds_tp1, r=RISK_FREE)
        prices_hedge_tp1_option = _ds_price(day_df_tp1, spot_tp1, hc_ds_tp1, r=RISK_FREE)
        prices_hedge_t = assemble_total_vector(len(sorted_hedges), option_indices, prices_hedge_t_option, underlying_idx, spot_t)
        prices_hedge_tp1 = assemble_total_vector(len(sorted_hedges), option_indices, prices_hedge_tp1_option, underlying_idx, spot_tp1)
        V_tp1 = float(prices_target_tp1.sum())
        Z_unhedged.append(V_tp1 - V0_ds)

        tgt_deltas, _ = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        phi_delta_new = float(tgt_deltas.sum())
        psi_delta = Pi_delta - phi_delta_new * spot_t
        Pi_delta_new = phi_delta_new * spot_tp1 + psi_delta * (1 + RISK_FREE / 252)
        Z_delta.append(V_tp1 - Pi_delta_new)
        Pi_delta = Pi_delta_new

        tgt_deltas_dv, tgt_vegas_dv = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        hdg_deltas_option, hdg_vegas_option = _ds_greeks(day_df_t, spot_t, hc_ds_t, r=RISK_FREE)
        hdg_deltas_dv, hdg_vegas_dv = assemble_total_greeks(len(sorted_hedges), option_indices, hdg_deltas_option, hdg_vegas_option, underlying_idx)
        kappa_h = float(hdg_vegas_dv[atm_idx])
        if abs(kappa_h) <= BENCHMARK_VEGA_FLOOR:
            return None
        phi_vega_new = float(tgt_vegas_dv.sum()) / kappa_h
        phi_delta_under = float(tgt_deltas_dv.sum()) - phi_vega_new * float(hdg_deltas_dv[atm_idx])
        trade_cost_dv = float(c_t0[atm_idx] * abs(phi_vega_new - phi_vega_atm))
        psi_dv = Pi_dv - phi_vega_new * prices_hedge_t[atm_idx] - phi_delta_under * spot_t - trade_cost_dv
        Pi_dv_new = phi_vega_new * prices_hedge_tp1[atm_idx] + phi_delta_under * spot_tp1 + psi_dv * (1 + RISK_FREE / 252)
        Z_dv.append(V_tp1 - Pi_dv_new)
        Pi_dv = Pi_dv_new
        phi_vega_atm = phi_vega_new

        tc_nw_t = _set_tau(target_contracts, tau_t)
        hc_nw_t = _set_tau(hedge_contracts, tau_t)
        tc_nw_next = _set_tau(target_contracts, tau_tp1)
        hc_nw_next = _set_tau(hedge_contracts, tau_tp1)
        prices_target_t_nw = bs_price_from_surface(log_iv_t, spot_t, tc_nw_t, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order)
        prices_hedge_t_nw_option = bs_price_from_surface(log_iv_t, spot_t, hc_nw_t, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order)
        scenarios_t = sample_diffusion_scenarios(ctx, date_t, spot_t, n_scenarios, batch_size)
        if scenarios_t is None:
            break
        spots_next, iv_next = scenarios_t
        dV_t, dH_t_option = scenarios_to_solver_arrays(spots_next, iv_next, spot_t, tc_nw_next, hc_nw_next, prices_target_t_nw, prices_hedge_t_nw_option, r=RISK_FREE, m_grid=m_grid, tau_grid=tau_grid)
        dH_t = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_t_option, underlying_idx, spots_next, spot_t)
        result = solve_transaction_cost_lasso(dV_t, dH_t, phi_diffusion, c_t0, alpha=alpha_best, g0_scale=V0)
        phi_new = result.phi
        trade_cost = float(np.dot(c_t0, np.abs(result.trade)))
        psi = Pi_diffusion - float(np.dot(phi_new, prices_hedge_t)) - trade_cost
        Pi_diffusion_new = float(np.dot(phi_new, prices_hedge_tp1)) + psi * (1 + RISK_FREE / 252)
        Z_diffusion.append(V_tp1 - Pi_diffusion_new)
        phi_diffusion = phi_new
        Pi_diffusion = Pi_diffusion_new

    if not Z_diffusion:
        return None
    return {"unhedged": Z_unhedged, "delta": Z_delta, "delta_vega": Z_dv, "diffusion": Z_diffusion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--m0", type=float, default=1.0)
    parser.add_argument("--all-m0", action="store_true")
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-start", default="2018-07-01")
    parser.add_argument("--test-end", default="2023-02-28")
    parser.add_argument("--max-windows", type=int, default=52)
    parser.add_argument("--exclude-covid", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ctx = load_diffusion_context(args.checkpoint, args.train_data, args.processed_dir, args.device)
    state_lookup = build_state_lookup(args.prepared_dir)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    delta_surface_lookup = load_delta_surface(args.data_dir, start_year=test_start.year - 1, end_year=test_end.year)

    m0_values = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05] if args.all_m0 else [args.m0]
    monthly_starts = pd.date_range(test_start, test_end, freq="MS")
    covid_start = pd.Timestamp("2020-02-13")
    covid_end = pd.Timestamp("2020-07-21")
    results_all = {"unhedged": [], "delta": [], "delta_vega": [], "diffusion": []}
    n_windows = 0

    for m0 in m0_values:
        if args.all_m0:
            print(f"\nm0 = {m0}")
        for candidate in monthly_starts:
            if not args.all_m0 and n_windows >= args.max_windows:
                break
            if args.exclude_covid and covid_start <= candidate <= covid_end:
                continue
            label = f"m0={m0} {candidate.date()}" if args.all_m0 else f"{candidate.date()}"
            print(f"  {label} ...", end=" ", flush=True)
            try:
                panel = build_instrument_panel(candidate, m0=m0, data_dir=args.data_dir)
            except Exception as exc:
                print(f"SKIP (panel: {exc})")
                continue
            window_results = run_one_window(panel, ctx, state_lookup, args.n_scenarios, args.n_val, args.batch_size, delta_surface_lookup)
            if window_results is None:
                print("SKIP (insufficient data)")
                continue
            for method in results_all:
                results_all[method].extend(window_results[method])
            print(f"OK ({len(window_results['diffusion'])} days, Z_diffusion std={np.std(window_results['diffusion']):.3f})")
            n_windows += 1

    print(f"\nTotal windows: {n_windows}, total observations: {len(results_all['diffusion'])}")
    if not results_all["diffusion"]:
        return
    print_table2(results_all)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"method": method, **tracking_error_stats(Z)} for method, Z in results_all.items()]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        if len({len(Z) for Z in results_all.values()}) == 1:
            raw = pd.DataFrame({method: np.asarray(Z, dtype=float) for method, Z in results_all.items()})
            raw.insert(0, "observation", np.arange(len(raw)))
            raw_path = args.output.with_name(f"{args.output.stem}_raw{args.output.suffix}")
            raw.to_csv(raw_path, index=False)
            print(f"Raw tracking errors saved to {raw_path}")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
