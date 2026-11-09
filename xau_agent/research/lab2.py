"""Banc d'essai multi-marchés : stratégies liées aux horaires de bourse + tendance H4 de référence.

Données : OANDA 5 min (huggingface.co/datasets/bunqi/OANDA_*), 2015-01 → 2026-10, fichiers <SYM>_<AAAAMM>.parquet.
Sélection sur 2015-2021 (IS) : espérance > 0 et t-stat >= 2 ; verdict sur 2022-2026 (OOS), jamais utilisé pour choisir.
Coûts : spread+commission en bp du prix (+0,10 $ sur l'or), swap facturé chaque nuit (sens défavorable, prudent).
Usage: python lab2.py <dossier_parquet>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import lab

#          fuseau,              ouverture, clôture, coût bp, swap bp/nuit, indice actions
MARKETS = {
    "SPX500": ("America/New_York", "09:30", "16:00", 1.0, 2.0, True),
    "NAS100": ("America/New_York", "09:30", "16:00", 1.0, 2.0, True),
    "US30":   ("America/New_York", "09:30", "16:00", 1.0, 2.0, True),
    "DE30":   ("Europe/Berlin",    "09:00", "17:30", 1.0, 2.0, True),
    "XAUUSD": ("Europe/London",    "08:00", "16:30", 1.0, 1.5, False),
    "EURUSD": ("Europe/London",    "08:00", "16:30", 0.7, 0.5, False),
    "WTI":    ("America/New_York", "09:00", "14:30", 1.0, 2.0, False),
}
BIG = 1e9
IS = ("2015-01-01", "2021-12-31")
OOS = ("2022-01-01", "2026-10-31")


def hm(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m


def load(folder, sym, tz, t_open, t_close):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{folder}/{sym}_*.parquet")))
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    df = df[~df.index.duplicated()].sort_index()[["open", "high", "low", "close"]]
    loc = df.index.tz_convert(tz)
    df["m"] = loc.hour * 60 + loc.minute
    df["ld"] = loc.tz_localize(None).normalize()
    df["dow"] = loc.dayofweek
    o, c = hm(t_open), hm(t_close)
    sess = df[(df.m >= o) & (df.m < c)]
    g = sess.groupby("ld")
    day = pd.DataFrame({"so": g.open.first(), "sh": g.high.max(), "sl": g.low.min(), "sc": g.close.last()})
    rng = df[(df.m >= o) & (df.m < o + 30)].groupby("ld")
    day["rh"], day["rl"] = rng.high.max(), rng.low.min()
    day["p30"] = df[df.m == o + 25].groupby("ld").close.last()       # prix à ouverture + 30 min
    day["prev_sc"] = day.sc.shift(1)
    tr = pd.concat([day.sh - day.sl, (day.sh - day.prev_sc).abs(), (day.sl - day.prev_sc).abs()], axis=1).max(axis=1)
    day["datr"] = tr.rolling(14).mean().shift(1)                       # ATR journalier connu la veille
    # calendrier : dernier jour de bourse du mois, 3e jour de bourse du mois
    d = day.index.to_series()
    ym = d.dt.to_period("M")
    day["last_td"] = d.groupby(ym).transform("max") == d
    day["td_rank"] = d.groupby(ym).cumcount() + 1
    df = df.join(day, on="ld")
    df["last_td"] = df.last_td.fillna(False).astype(bool)
    return df, o, c


def finish(df, side, sld, tpd, exitm, cost_bp, swap_bp, fix):
    side = np.where(np.isfinite(sld) & (sld > 0), np.nan_to_num(side), 0)
    tr = lab.simulate(df, side.astype(int), np.asarray(sld, float), np.asarray(tpd, float),
                      np.asarray(exitm, bool), cost_bp * 1e-4, fix)
    nights = ((tr.exit - pd.Timedelta(hours=22)).dt.normalize() - (tr.entry - pd.Timedelta(hours=22)).dt.normalize()).dt.days
    tr["R"] -= swap_bp * 1e-4 * tr.px * nights / tr.risk
    return tr


def strategies(df, o, c, is_index):
    m, cl = df.m.values, df.close.values
    in_sess = (m >= o) & (m < c)
    eod = (m >= c - 5) | (m < o)
    fri_late = (df.dow.values == 4) & (m >= c - 5)
    out = {}

    # 1) Cassure du range des 30 premières minutes (ORB), 1re cassure dans les 2 h suivantes
    win = (m >= o + 30) & (m < o + 150)
    raw = np.where(win & (cl > df.rh.values), 1, np.where(win & (cl < df.rl.values), -1, 0))
    first = pd.Series(raw != 0).groupby(df.ld.values).cumsum().values == 1
    side = np.where(first, raw, 0)
    sl = np.where(side > 0, cl - df.rl.values, df.rh.values - cl)
    out["ORB 30min RR2"] = (side, sl, 2 * sl, eod)
    out["ORB 30min fin de séance"] = (side, sl, np.full(len(df), BIG), eod)

    # 2) Momentum intraday (Gao et al. 2018) : sens (veille clôture -> ouverture+30min) joué sur les 30 dernières min
    sig = m == c - 35
    mom = np.sign(df.p30.values - df.prev_sc.values)
    out["Momentum dernière 1/2h"] = (np.where(sig, mom, 0), 0.5 * df.datr.values, np.full(len(df), BIG), eod)

    # 3) Tendance H4 (référence) calculée à part
    if is_index:
        # 4) Comblement de gap à l'ouverture
        gap = df.so.values - df.prev_sc.values
        sig = (m == o) & (np.abs(gap) > 0.25 * df.datr.values)
        tp = np.abs(cl - df.prev_sc.values)
        side = np.where(sig & (np.sign(cl - df.prev_sc.values) == np.sign(gap)), -np.sign(gap), 0)
        out["Comblement de gap"] = (side, np.abs(gap), tp, eod)
        # 5) Overnight : achat à la clôture, revente à l'ouverture (pas le vendredi)
        sig = (m == c - 10) & (df.dow.values < 4)
        out["Overnight acheteur"] = (np.where(sig, 1, 0), df.datr.values, np.full(len(df), BIG), m == o)
        # 6) Effet de fin de mois : achat dernier jour du mois, revente 3e jour du mois suivant (garde le week-end)
        sig = (m == c - 10) & df.last_td.values
        ex = (df.td_rank.values == 3) & (m >= c - 5) & ~df.last_td.values
        out["Fin de mois acheteur (swing)"] = (np.where(sig, 1, 0), 2 * df.datr.values, np.full(len(df), BIG), ex)
    return out


def h4_trend(df, sym, cost_bp, swap_bp, fix):
    x = df[["open", "high", "low", "close"]].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    p = f"/tmp/_h4_{sym}.csv"
    x.to_csv(p)
    h = lab.prepare(p)
    os.remove(p)
    h["session"] = True
    side, sl, tp, ex = lab.trend_breakout(h, 20, 2.0, 4.0)
    return finish(h, side, sl, tp, ex, cost_bp, swap_bp, fix)


def period(tr, a, b):
    return tr[(tr.entry >= pd.Timestamp(a, tz="UTC")) & (tr.entry <= pd.Timestamp(b, tz="UTC"))].reset_index(drop=True)


def tstat(r):
    return r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 2 else np.nan


def main():
    folder = sys.argv[1]
    rows, all_tr = [], {}
    for sym, (tz, to, tc, cb, sw, idx) in MARKETS.items():
        df, o, c = load(folder, sym, tz, to, tc)
        fix = 0.10 if sym == "XAUUSD" else 0.0
        strats = {k: finish(df, *v, cb, sw, fix) for k, v in strategies(df, o, c, idx).items()}
        strats["Tendance H4 Donchian20"] = h4_trend(df, sym, cb, sw, fix)
        for name, tr in strats.items():
            a, b = period(tr, *IS), period(tr, *OOS)
            all_tr[(sym, name)] = tr
            rows.append(dict(marche=sym, strategie=name, trades_an=round(len(a) / 7),
                             espR_IS=round(a.R.mean(), 3), t_IS=round(tstat(a.R), 1),
                             espR_OOS=round(b.R.mean(), 3), t_OOS=round(tstat(b.R), 1),
                             PF_OOS=round(b.R[b.R > 0].sum() / -b.R[b.R < 0].sum(), 2) if len(b) else np.nan))
        print(sym, "ok", file=sys.stderr)
    res = pd.DataFrame(rows)
    res["retenu_IS"] = (res.espR_IS > 0) & (res.t_IS >= 2) & (res.trades_an >= 10)
    pd.set_option("display.width", 220)
    print(res.sort_values("t_IS", ascending=False).to_string(index=False))
    res.to_csv("lab2_results.csv", index=False)

    sel = res[res.retenu_IS]
    print(f"\n{len(sel)} stratégies retenues sur {len(res)} testées (sélection IS uniquement)")
    if len(sel) == 0:
        return
    port = pd.concat([all_tr[(r.marche, r.strategie)] for r in sel.itertuples()]).sort_values("entry").reset_index(drop=True)
    for per, (a, b) in (("IS", IS), ("OOS", OOS)):
        p = period(port, a, b)
        print(f"\nPortefeuille {per}: {len(p)} trades, {len(p) / ((pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25):.0f}/an,"
              f" espR {p.R.mean():.3f}, t {tstat(p.R):.1f}")
        for risk in (0.5, 1.0):
            pr, dur, n = lab.ftmo(p, pd.date_range(a, b, freq="W-MON", tz="UTC"), risk=risk, daily_stop=max(2, 2 * risk))
            print(f"  FTMO 2 phases, risque {risk}%: réussite {pr:.0f}% | durée médiane {dur} j | {n} challenges terminés")
    p = period(port, *OOS)
    print("\nR par année (OOS):", p.groupby(p.entry.dt.year).R.sum().round(1).to_dict())


if __name__ == "__main__":
    main()
