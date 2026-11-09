"""Convertit les CSV M1 HistData (heure EST fixe, UTC-5) en M15 UTC.
Source utilisée : huggingface.co/datasets/fokan/xauusd-2009-2026 (DAT_MT_XAUUSD_M1_<année>.csv)
Usage: python prep_data.py <dossier_m1> <sortie.csv>
"""
import glob
import sys

import pandas as pd

files = sorted(glob.glob(f"{sys.argv[1]}/*.csv"))
df = pd.concat(pd.read_csv(f, header=None, names=["d", "t", "open", "high", "low", "close", "v"]) for f in files)
df["time"] = pd.to_datetime(df["d"] + " " + df["t"], format="%Y.%m.%d %H:%M") + pd.Timedelta(hours=5)
df = df.set_index("time").sort_index()
m15 = df.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
m15.index = m15.index.tz_localize("UTC")
m15.to_csv(sys.argv[2])
print(len(m15), m15.index[0], m15.index[-1])
