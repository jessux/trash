"""Banc d'essai : compare des agents candidats sur XAUUSD M15 et simule des challenges FTMO.

Protocole anti-surapprentissage :
- sélection uniquement sur 2015-2021 (in-sample), verdict sur 2022-2025 (out-of-sample) ;
- agent IA (gradient boosting) entraîné en walk-forward : pour l'année Y, entraîné seulement sur les années < Y.
Hypothèses : entrée à l'ouverture de la bougie suivante, coût = 1 bp du prix + 0.10$ (spread+commission+slippage),
SL avant TP si les deux sont touchés dans la même bougie, gap au-delà du SL exécuté à l'ouverture,
flat vendredi 19h UTC.
Usage: python lab.py xau_m15.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

IS_END, OOS_START = "2021-12-31", "2022-01-01"


# ---------- indicateurs ----------
def prepare(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    c = df["close"]
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema800"] = c.ewm(span=800, adjust=False).mean()
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    for n in (20, 55):
        df[f"dch{n}"] = df.high.rolling(n).max().shift(1)
        df[f"dcl{n}"] = df.low.rolling(n).min().shift(1)
    h = df.index.hour
    df["hour"], df["dow"] = h, df.index.dayofweek
    df["session"] = (h >= 7) & (h < 19)
    df["flat"] = (df.dow == 4) & (h >= 19)
    return df


# ---------- agents stratégie : renvoient (side, sl_dist, tp_dist, exit_mask) ----------
def trend_breakout(df, dc=20, sl=1.5, tp=3.0):
    c = df.close
    side = np.where((c > df.ema200) & (c > df[f"dch{dc}"]), 1, np.where((c < df.ema200) & (c < df[f"dcl{dc}"]), -1, 0))
    side = np.where(df.session & ~df.flat, side, 0)
    return side, sl * df.atr.values, tp * df.atr.values, df.flat.values


def london_orb(df, rr=1.5):
    day = df.index.date
    asia = df[df.hour < 7]
    rh = asia.groupby(asia.index.date).high.max()
    rl = asia.groupby(asia.index.date).low.min()
    hi = pd.Series(day, index=df.index).map(rh).values
    lo = pd.Series(day, index=df.index).map(rl).values
    win = (df.hour >= 7) & (df.hour < 11) & ~df.flat
    raw = np.where(win & (df.close > hi), 1, np.where(win & (df.close < lo), -1, 0))
    # uniquement la 1re cassure du jour
    first = pd.Series(raw != 0, index=df.index).groupby(day).cumsum().values == 1
    side = np.where(first & (raw != 0), raw, 0)
    width = hi - lo
    ok = width < 4 * df.atr.values * 4  # range asiatique pas aberrant
    side = np.where(ok, side, 0)
    sl = np.where(side > 0, df.close.values - lo, hi - df.close.values)
    return side, sl, rr * sl, (df.hour >= 20).values | df.flat.values


def trend_pullback(df, sl=2.0, tp=2.0):
    c = df.close
    side = np.where((c > df.ema800) & (df.rsi < 30), 1, np.where((c < df.ema800) & (df.rsi > 70), -1, 0))
    side = np.where(df.session & ~df.flat, side, 0)
    return side, sl * df.atr.values, tp * df.atr.values, df.flat.values


# ---------- simulateur de trades ----------
def simulate(df, side, sld, tpd, exitm, cost_bp=1e-4, cost_fix=0.10):
    t, o, h, l, c = df.index, df.open.values, df.high.values, df.low.values, df.close.values
    trades, pos, pending = [], None, None
    for j in range(len(df)):
        if pending is not None:
            s, a, b, i0 = pending
            e = o[j]
            pos, pending = (s, e, e - s * a, e + s * b, a, j, i0), None
        if pos:
            s, e, slp, tpp, a, j0, i0 = pos
            gap = j > j0 and ((s > 0 and o[j] <= slp) or (s < 0 and o[j] >= slp))
            hit_sl = (l[j] <= slp) if s > 0 else (h[j] >= slp)
            hit_tp = (h[j] >= tpp) if s > 0 else (l[j] <= tpp)
            x = o[j] if gap else slp if hit_sl else tpp if hit_tp else c[j] if exitm[j] else None
            if x is None:
                continue
            cost = cost_bp * e + cost_fix
            trades.append((t[i0], t[j0], t[j], s, ((x - e) * s - cost) / a, e, a))
            pos = None
            continue
        if side[j] != 0 and np.isfinite(sld[j]) and sld[j] > 0 and j + 1 < len(df):
            pending = (side[j], sld[j], tpd[j], j)
    return pd.DataFrame(trades, columns=["signal", "entry", "exit", "side", "R", "px", "risk"])


# ---------- agent IA : filtre ML walk-forward (meta-labeling) ----------
FEATS = ["ret4", "ret16", "ret96", "atr_rel", "dist_ema200", "dist_ema800", "rsi", "hour", "dow", "side"]


def ml_filter(df, trades):
    f = pd.DataFrame(index=df.index)
    for n in (4, 16, 96):
        f[f"ret{n}"] = df.close.pct_change(n)
    f["atr_rel"] = df.atr / df.atr.rolling(960).mean()
    f["dist_ema200"] = (df.close - df.ema200) / df.atr
    f["dist_ema800"] = (df.close - df.ema800) / df.atr
    f["rsi"], f["hour"], f["dow"] = df.rsi, df.hour, df.dow
    X = f.loc[trades.signal].reset_index(drop=True)
    X["side"] = trades.side.values
    y = (trades.R > 0).astype(int).values
    yr = trades.signal.dt.year.values
    keep = np.zeros(len(trades), bool)
    for Y in range(2018, yr.max() + 1):  # >= 3 ans d'historique pour entraîner
        tr, te = yr < Y, yr == Y
        if te.sum() == 0:
            continue
        # on n'entraîne que sur des trades CLOTURES avant le début de l'année Y
        tr &= (trades.exit < pd.Timestamp(f"{Y}-01-01", tz="UTC")).values
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
        m.fit(X[tr][FEATS], y[tr])
        keep[te] = m.predict_proba(X[te][FEATS])[:, 1] > y[tr].mean()
    return trades[keep & (yr >= 2018)].reset_index(drop=True)


# ---------- simulation de challenges FTMO ----------
def ftmo(trades, starts, risk=0.5, daily_stop=2.0, max_stop=6.0, max_day=2, max_consec=2):
    entry = trades.entry.values
    days = trades.entry.dt.tz_convert("Europe/Prague").dt.date.values
    R = trades.R.values

    def phase(i, target):
        eq, day, d0, n, consec, tdays = 0.0, None, 0.0, 0, 0, set()
        while i < len(R):
            if days[i] != day:
                day, d0, n, consec = days[i], eq, 0, 0
            room = daily_stop - (d0 - eq)
            if n < max_day and consec < max_consec and room > 0.05:
                r = min(risk, room, eq + max_stop)
                eq += R[i] * r
                n += 1
                consec = consec + 1 if R[i] < 0 else 0
                tdays.add(day)
                if eq <= -max_stop:
                    return False, i
                if eq >= target:
                    return True, i
            i += 1
        return None, i  # données épuisées

    out = []
    for s in starts:
        i = int(np.searchsorted(entry, np.datetime64(s.tz_convert(None))))
        p1, j = phase(i, 10.0)
        p2, k = phase(j + 1, 5.0) if p1 else (None, j)
        if p1 is None or (p1 and p2 is None):
            continue  # challenge non terminé faute de données : exclu
        dur = (trades.exit.iloc[k] - s).days
        out.append((bool(p1 and p2), dur))
    o = pd.DataFrame(out, columns=["ok", "days"])
    return o.ok.mean() * 100 if len(o) else np.nan, o[o.ok].days.median() if o.ok.any() else np.nan, len(o)


def stats(tr):
    if len(tr) == 0:
        return dict(n=0)
    r = tr.R
    eq = r.cumsum()
    yrs = (tr.entry.iloc[-1] - tr.entry.iloc[0]).days / 365.25
    return dict(n=len(r), trades_an=round(len(r) / yrs), winrate=round((r > 0).mean() * 100),
                espR=round(r.mean(), 3), PF=round(r[r > 0].sum() / -r[r < 0].sum(), 2),
                maxDD_R=round((eq.cummax() - eq).max(), 1))


def main():
    df = prepare(sys.argv[1])
    agents = {}
    for dc in (20, 55):
        for tp in (2.0, 3.0):
            agents[f"trend dc{dc} tp{tp}"] = simulate(df, *trend_breakout(df, dc, 1.5, tp))
    for rr in (1.0, 1.5, 2.0):
        agents[f"orb rr{rr}"] = simulate(df, *london_orb(df, rr))
    for sl, tp in ((2.0, 2.0), (1.5, 3.0)):
        agents[f"pullback sl{sl} tp{tp}"] = simulate(df, *trend_pullback(df, sl, tp))
    for base in ("trend dc20 tp3.0", "orb rr1.5", "pullback sl2.0 tp2.0"):
        agents[f"IA+{base}"] = ml_filter(df, agents[base])

    rows = []
    for name, tr in agents.items():
        for per, a, b in (("IS", "2018-01-01", IS_END), ("OOS", OOS_START, "2026-01-01")):
            # IS commence en 2018 pour tous, afin de comparer à armes égales avec l'IA walk-forward
            sub = tr[(tr.entry >= pd.Timestamp(a, tz="UTC")) & (tr.entry <= pd.Timestamp(b, tz="UTC"))].reset_index(drop=True)
            starts = pd.date_range(a, b, freq="W-MON", tz="UTC")
            pr, dur, nch = ftmo(sub, starts) if len(sub) else (np.nan, np.nan, 0)
            rows.append(dict(agent=name, periode=per, **stats(sub), ftmo_reussite_pct=round(pr, 1),
                             jours_median=dur, challenges=nch))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for per in ("IS", "OOS"):
        print(f"\n=== {per} ===")
        print(res[res.periode == per].drop(columns="periode").sort_values("espR", ascending=False).to_string(index=False))
    res.to_csv("lab_results.csv", index=False)
    # détail par année du meilleur IS
    best = res[res.periode == "IS"].sort_values("ftmo_reussite_pct", ascending=False).agent.iloc[0]
    tr = agents[best]
    print(f"\nMeilleur en IS: {best}\nEspérance R par année:")
    print(tr.groupby(tr.entry.dt.year).R.agg(["count", "mean", "sum"]).round(3).to_string())


if __name__ == "__main__":
    main()
