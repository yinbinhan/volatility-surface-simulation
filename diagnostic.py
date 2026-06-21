import sys, types, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "../VolGAN")

# shims
_stub = types.ModuleType("pandas_datareader")
_stub.data = types.ModuleType("pandas_datareader.data")
sys.modules["pandas_datareader"] = _stub
sys.modules["pandas_datareader.data"] = _stub.data
import scipy as _scipy
if not hasattr(_scipy, "arange"):
    _scipy.arange = np.arange; _scipy.array = np.array; _scipy.exp = np.exp

from hedging import build_instrument_panel
from volgan_adapter import MONEYNESS_GRID, TAU_GRID

# Build one panel
panel = build_instrument_panel(
    "2018-07-01", m0=0.9,
    data_dir=Path("data/VolGAN_optionmetrics_spx_20000103_20230228")
)
t0 = panel.trading_dates[0]
print(f"t0={t0}  n_trading_days={len(panel.trading_dates)}")
print(f"target optionids: {list(panel.target['optionid'])}")
print(f"n_hedges: {len(panel.hedges)}")

# Check quotes coverage
print(f"\nquotes date dtype: {panel.quotes['date'].dtype}")
print(f"quotes date range: {panel.quotes['date'].min()} to {panel.quotes['date'].max()}")
print(f"unique dates in quotes: {panel.quotes['date'].nunique()}")
print(f"quotes on t0 ({t0}): {len(panel.quotes[panel.quotes['date'] == t0])} rows")
print(f"quotes on t0+1 ({panel.trading_dates[1]}): {len(panel.quotes[panel.quotes['date'] == panel.trading_dates[1]])} rows")

# Check date_to_idx
df = pd.read_csv("data/volgan_prepared/dates.csv", parse_dates=["date"])
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
print(f"\nt0 in date_to_idx: {t0 in date_to_idx}")
print(f"t0 type: {type(t0)}, key example type: {type(list(date_to_idx.keys())[0])}")
