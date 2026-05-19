# VolGAN Paper-Faithful Hedging Layer (Port)

## Purpose

This subdirectory contains an **unmodified copy** of the paper-faithful
hedging components from the sibling `VolGAN` repo, ported into the
diffusion-model repo so that the diffusion-driven scenario producer can
drive the same hedging layer used in the VolGAN experiments.

The files here are intended as a **drop-in scenario adapter contract**:
the diffusion sampler emits scenarios in the shared-grid format
(strike/maturity grid, IV channel layout) defined by
`shared_grid_preprocessing.py`, and `hedging.py` / `volgan_experiment.py`
consume them through the same writer + adapter glue that the VolGAN
generator uses.

## Source provenance

- Source repository: `/home/yinbinha/VolGAN`
- Source repository HEAD at port time: `be662ac` (Add shared-grid VolGAN data loader)
- Source files were uncommitted modifications on top of that HEAD; the
  authoritative fingerprint is the md5 listed below.
- Port branch: `volgan-hedging-port-20260519` in
  `/home/yinbinha/volatility-surface-simulation`.
- Port date: 2026-05-19.
- Checkpoint reference: VolGAN `DS 20260519_0155_bw` (paper-faithful bandwidths
  h1 = 0.002, h2 = 0.046).

## Files

| Destination (this dir)                          | Source under `/home/yinbinha/VolGAN/`     | md5                                |
|------------------------------------------------ |------------------------------------------ |----------------------------------- |
| `hedging.py`                                    | `hedging.py`                              | `d8914f0124c3af8455a13208fe12d763` |
| `volgan_experiment.py`                          | `volgan_experiment.py`                    | `f7e2a816565dcd1a660ac89a5880c5b2` |
| `shared_grid_preprocessing.py`                  | `shared_grid_preprocessing.py`            | `90426a3171d472d071cb534795b61d48` |
| `VOLGAN_AGENTS.md`                              | `AGENTS.md`                               | `e6a3329d9558ea30d04cb2fcbcc08540` |
| `paper_context/HEDGING_2025_CONTEXT.md`         | `paper_context/HEDGING_2025_CONTEXT.md`   | `433723e02a7a48a6bbb927e323e57c19` |
| `paper_context/HEDGING_PHASE_MONITOR.md`        | `paper_context/HEDGING_PHASE_MONITOR.md`  | `d94a3f1123db7dc3d7b90e0d2ab5f1ca` |
| `paper_context/VOLGAN_2024_CONTEXT.md`          | `paper_context/VOLGAN_2024_CONTEXT.md`    | `02ceb148c5c2509fb448b64269153fa4` |
| `paper_context/VOLGAN_EXPERIMENT_2026.md`       | `paper_context/VOLGAN_EXPERIMENT_2026.md` | `3f2ae1f0de5e43198d2ee471082e06ed` |

`AGENTS.md` from the VolGAN repo is renamed `VOLGAN_AGENTS.md` here to
avoid colliding with this repo's own (untracked) `AGENTS.md`.

## Rules for this subdirectory

- **Do not edit these files in place.** They are the canonical
  paper-faithful artifacts; if a divergence is needed for the diffusion
  driver, create a new file (e.g. `hedging_diffusion_adapter.py`) that
  wraps or subclasses, and re-document the divergence here.
- **Do not overwrite the existing repo-root files** (`hedging.py`,
  `shared_grid_preprocessing.py`, `prepare_shared_grid_data.py`,
  `SHARED_GRID_HEDGING_WORKFLOW.md`). Those are older codex-authored
  versions kept for reference and are intentionally untouched by this
  port.
- If the VolGAN source changes, re-port by md5: bump the table above
  and the commit message; do not silently edit.

## Intended use

The diffusion scenario producer should:

1. Emit shared-grid scenarios matching the layout produced by
   `shared_grid_preprocessing.py` (paper-faithful bandwidths
   h1 = 0.002, h2 = 0.046).
2. Feed those scenarios into the writer/adapter glue in
   `volgan_experiment.py`, which then calls into `hedging.py`.
3. Compare hedging PnL / risk metrics against the VolGAN baseline using
   identical hedging logic.
