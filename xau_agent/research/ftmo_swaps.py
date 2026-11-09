"""Récupère les swaps FTMO publiés (ftmo.com/wp-json/ftmo/symbols) et les convertit en bp du prix par nuit
(signe + = crédit, − = coût), avec le dernier prix des données journalières OANDA.
Usage: python ftmo_swaps.py <dossier_journalier> [sym.json]  -> écrit ftmo_swaps.csv
"""
import glob
import json
import sys
import urllib.request

import pandas as pd

MAP = {  # symbole OANDA -> (code FTMO, facteur d'unité OANDA -> FTMO)
    "SPX500USD": ("US500.cash", 1), "NAS100USD": ("US100.cash", 1), "US2000USD": ("US2000.cash", 1),
    "DE30EUR": ("GER40.cash", 1), "UK100GBP": ("UK100.cash", 1), "EU50EUR": ("EU50.cash", 1),
    "JP225USD": ("JP225.cash", 1), "AU200AUD": ("AUS200.cash", 1), "HK33HKD": ("HK50.cash", 1),
    "EURUSD": ("EUR/USD", 1), "GBPUSD": ("GBP/USD", 1), "USDJPY": ("USD/JPY", 1), "AUDUSD": ("AUD/USD", 1),
    "USDCAD": ("USD/CAD", 1), "USDCHF": ("USD/CHF", 1), "NZDUSD": ("NZD/USD", 1),
    "XAUUSD": ("XAU/USD", 1), "XAGUSD": ("XAG/USD", 1), "XPTUSD": ("XPT/USD", 1), "XPDUSD": ("XPD/USD", 1),
    "WTICOUSD": ("USOIL.cash", 1), "BCOUSD": ("UKOIL.cash", 1), "NATGASUSD": ("NATGAS.cash", 1),
    "CORNUSD": ("CORN.c", 100), "WHEATUSD": ("WHEAT.c", 100), "SOYBNUSD": ("SOYBEAN.c", 100),  # FTMO en cents
    "SUGARUSD": ("SUGAR.c", 100),
}

folder = sys.argv[1]
if len(sys.argv) > 2:
    raw = json.load(open(sys.argv[2]))
else:
    raw = json.load(urllib.request.urlopen("https://ftmo.com/wp-json/ftmo/symbols", timeout=30))
ftmo = {s["code"]: s for s in raw["data"]["symbols"] if s["active"]}
# toutes les paires forex FTMO présentes dans le dossier (EURGBP -> EUR/GBP)
for f in glob.glob(f"{folder}/*_*.parquet"):
    p = f.split("/")[-1].split("_")[0]
    if len(p) == 6 and f"{p[:3]}/{p[3:]}" in ftmo:
        MAP.setdefault(p, (f"{p[:3]}/{p[3:]}", 1))
rows = []
for sym, (code, unit) in MAP.items():
    s = ftmo[code]
    x = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{folder}/{sym}_*.parquet")).sort_values("time")
    px = x.close.iloc[-1] * unit
    assert s["swapType"] == "points", code
    pt = 10 ** -s["digits"]
    rows.append(dict(oanda=sym, ftmo=code, prix=round(px, 4), swap_long_pts=s["swapLong"], swap_short_pts=s["swapShort"],
                     long_bp=round(s["swapLong"] * pt / px * 1e4, 2), short_bp=round(s["swapShort"] * pt / px * 1e4, 2)))
df = pd.DataFrame(rows)
df["long_pct_an"] = (df.long_bp * 365 / 100).round(1)
df["short_pct_an"] = (df.short_bp * 365 / 100).round(1)
df.to_csv("ftmo_swaps.csv", index=False)
print(df.to_string(index=False))
