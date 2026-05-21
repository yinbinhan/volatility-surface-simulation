# Hedging code changes — 2026-05-18

All changes are in `volatility-surface-simulation/hedging.py`.
Verified: all four self-checks pass after changes.

---

## 1. LASSO objective — paper eq. (20)

**What changed:** Three bugs fixed in `solve_transaction_cost_lasso` and `_transaction_cost_lasso_components`.

| Bug | Before | After | Justification |
|-----|--------|-------|---------------|
| Normalization | `0.5 * mean(residuals²)` | `mean(residuals²)` | Paper uses `(1/N)Σ[…]²`, no 0.5 factor. `lambdas` adjusted to `0.5 * alpha * g0_scale * costs` so coordinate-descent updates stay consistent. |
| g₀ scaling | Missing entirely | `alpha * g0_scale * sum(c_i * |trade_i|)` | Paper §3: penalty scaled by g₀ = V_t (current portfolio value) so alpha is dimensionless regardless of portfolio size. `g0_scale` computed as `sum(current_target_mid)` each day. |
| A_t intercept | Fixed constant `g0=0.0` | Jointly estimated in coordinate descent | Paper eq.(20): `A_t ∈ ℝ` is a free, unpenalized parameter. Now estimated each iteration as `intercept += mean(residual)`. |

`TransactionCostLassoResult`: renamed `g0` → `intercept` (A_t); added `g0_scale`.

---

## 2. AIC formula — paper eq. (24)

**What changed:** `select_alpha_aic` AIC computation.

| Bug | Before | After | Justification |
|-----|--------|-------|---------------|
| Intercept not counted | `2 * active_trades` | `2 * (1 + active_trades)` | Paper eq.(24): `+1` counts the unpenalized A_t as a free parameter. |
| Validation intercept | Used fixed `g0` | Re-estimates `A_val = mean(y_val - x_val @ phi)` | Paper eq.(24) RSS uses `Â₀(α)` evaluated on the validation set, not the training intercept. |

---

## 3. Tracking error — paper §2 eq. (6)

**What changed:** `DailyBacktestResult`, `_daily_result`, `tracking_error_summary`.

- Added `cumulative_tracking_error: float` to `DailyBacktestResult`. Accumulated as `prev_z + realized_te` each day; Z₀ = 0.
- `tracking_error_summary` now adds `terminal_*` columns (mean, std, VaR at 5/2.5/1%) computed over the last cumulative value per window — these are the paper's Table 2 statistics.

**Justification:** Paper defines Z_T = V_T − Π_T at expiry (cumulative), not per-day increments. Table 2 reports statistics over 52 terminal Z_T values.

---

## 4. run_daily_backtest — once-per-window alpha (paper §4.2)

**What changed:** New `fixed_alpha: float | None = None` parameter.

- When `fixed_alpha` is set: call `solve_transaction_cost_lasso` directly each day with the pre-selected alpha.
- When `None`: fall back to per-day 70/30 split (legacy mode, preserved for backward compat).

**Justification:** Paper §4.2 selects alpha once at t=0 using N=1000 fit + M=100 validation scenarios, then holds it fixed for all 21 rebalancing days. Per-day re-selection is not paper-compliant.

---

## 5. run_full_backtest — 52-window orchestrator (new function)

**What changed:** New `run_full_backtest(window_start_dates, m0_list, scenario_source, ...)` function.

- Loops over all (window_start, m0) pairs.
- At t=0: draws n_fit+n_val scenarios, selects alpha once via `select_alpha_aic`.
- Calls `run_daily_backtest` with `fixed_alpha`.
- Collects terminal Z_T (last cumulative value) per window.
- Returns a DataFrame with columns: `window_start`, `expiry`, `m0`, `strategy`, `terminal_Z_T`, `skipped_intervals`.

**Justification:** Paper runs 52 non-overlapping one-month windows. No outer loop existed before.

---

## 6. benchmark_hedge_positions — paper §4.4 ATM formula

**What changed:** Added optional `atm_optionid` parameter.

When `atm_optionid` is provided (paper-compliant path):
- `phi_vega = kappa_V / kappa_H` (portfolio vega / ATM option vega)
- `phi_delta = Delta_V − phi_vega * Delta_H` (residual delta)

Falls back to min-norm lstsq when `atm_optionid=None` (legacy, backward compatible).

**Justification:** Paper §4.4 specifies a single ATM option (K = S₀) for delta-vega hedging, not minimum-norm across all instruments. The min-norm approach uses more instruments and matches both greeks simultaneously, which is not the paper's benchmark.

---

## Remaining open items (not changed)

- `atm_optionid` for delta-vega benchmark needs to be wired into `run_daily_backtest` and `run_full_backtest` — currently the caller must pass it explicitly to `benchmark_hedge_positions`.
- The underlying (SPX index itself) is not in the hedge panel. Paper delta benchmark uses only the underlying; current code distributes delta residual across available options.
- `run_full_backtest` alpha-selection fallback uses the first daily scenario draw rather than a dedicated pre-window draw — scenario sources that don't support `alpha_scenarios_for_window` get a 70/30 split of the first interval's scenarios.

---

## Comparison against README "Remaining for full empirical results"

| README item | Status | Notes |
|-------------|--------|-------|
| Train production diffusion checkpoints | **Blocked** | Preprocessing pipeline (Steps 1–2) can run now that OptionMetrics data is available. Need checkpoint from PhD or a training run. |
| Export K one-day-ahead scenarios per hedge date | **Infrastructure done; blocked by checkpoint** | `IVSurfaceScenarios` adapter is wired into `run_full_backtest`. Missing: a `ScenarioSource` wrapper around the trained diffusion model. |
| Run full hedging evaluation over test period | **Driver done; blocked by checkpoint** | `run_full_backtest` implements the 52-window loop with paper-compliant alpha selection and terminal Z_T collection. Unblocked once a scenario source exists. |
| Compare diffusion vs VolGAN using same data grid and hedging protocol | **Blocked** | Needs (a) diffusion model scenario source and (b) VolGAN scenario source or pre-computed samples. The shared 11×9 grid and hedging protocol are already aligned. |

**Critical path:** get the diffusion model checkpoint → write a thin `DiffusionScenarioSource` → run `run_full_backtest`.
