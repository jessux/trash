"""Retour à la moyenne sur les 28 paires forex FTMO.

4 stratégies classiques, paramètres fixés AVANT le test (aucune optimisation), évaluées en portefeuille :
  J-RSI2   journalier : achat si RSI(2) < 10, vente si > 90 ; sortie quand RSI(2) repasse 50 ; stop 3 ATR ; max 10 j.
  J-BOLL   journalier : clôture sous Bollinger(20,2) basse -> achat, au-dessus haute -> vente ; sortie à la
           moyenne 20 ; stop 2 ATR ; max 20 j.
  H-ASIE   horaire, session asiatique : mêmes signaux Bollinger(20,2), entrées 23h-05h UTC seulement ;
           sortie à la moyenne 20 ou à 07h UTC ; stop 1,5 ATR.
  H-RSI    horaire : RSI(14) < 25 achat / > 75 vente ; sortie quand RSI repasse 50 ; stop 2 ATR ; max 24 h.
Entrée à l'ouverture de la bougie suivante, stop prioritaire, gap exécuté à l'ouverture, pas d'entrée le vendredi
après 20h UTC. Coûts : 0,8 bp (majeures) / 1,5 bp (croisées) aller-retour + swaps FTMO réels (ftmo_swaps.csv).
Sélection sur 2006-2015 (IS), verdict 2016-2026 (OOS).
FTMO : risque 0,5 % ou 1 %/trade, risque ouvert <= 3 %, stop journalier 2 %, stop interne -6 %, objectifs 10 % puis 5 %.
Usage: python lab4.py <dossier_journalier> <dossier_horaire>
"""
import glob
import sys

import numpy as np
import pandas as pd

MAJ = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
CRO = ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
       "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "NZDJPY", "NZDCAD", "NZDCHF", "CADJPY", "CADCHF", "CHFJPY"]
COST = {p: 0.8 for p in MAJ} | {p: 1.5 for p in CRO}
SWAP = {r.oanda: (r.long_bp, r.short_bp) for r in pd.read_csv("ftmo_swaps.csv").itertuples()}
IS, OOS = ("2006-06-01", "2015-12-31"), ("2016-01-01", "2026-10-31")


def load(folder, pair, daily):
    x = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{folder}/{pair}_*.parquet")))
    t = pd.to_datetime(x.pop("time"), unit="s", utc=True)
    if daily:  # bougie ouverte à 17h New York = séance du lendemain ; horodatage = 22h UTC veille environ
        t = t.dt.tz_convert(None).add(pd.Timedelta(hours=7)).dt.normalize().dt.tz_localize("UTC")
    x.index = t
    x = x[~x.index.duplicated(keep="last")].sort_index()[["open", "high", "low", "close"]]
    c = x.close
    tr = pd.concat([x.high - x.low, (x.high - c.shift()).abs(), (x.low - c.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    for n in (2, 14):
        d = c.diff()
        up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
        x[f"rsi{n}"] = 100 - 100 / (1 + up / dn)
    x["ma"] = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["bb_up"], x["bb_dn"] = x.ma + 2 * sd, x.ma - 2 * sd
    return x


def signals(x, name):
    c, h = x.close, x.index.hour
    block = (x.index.dayofweek == 4) & (h >= 20) if name.startswith("H") else x.index.dayofweek == 4
    none = np.zeros(len(x), bool)
    if name == "J-RSI2":
        return (x.rsi2 < 10), (x.rsi2 > 90), 3 * x.atr, (x.rsi2 > 50), (x.rsi2 < 50), 10, none, block
    if name == "J-BOLL":
        return (c < x.bb_dn), (c > x.bb_up), 2 * x.atr, (c >= x.ma), (c <= x.ma), 20, none, block
    if name == "H-ASIE":
        win = (h >= 23) | (h < 5)
        force = (h == 7)  # bougie 7h-8h UTC : sortie à sa clôture au plus tard
        return (c < x.bb_dn) & win, (c > x.bb_up) & win, 1.5 * x.atr, (c >= x.ma), (c <= x.ma), 9, np.asarray(force), block
    if name == "H-RSI":
        return (x.rsi14 < 25), (x.rsi14 > 75), 2 * x.atr, (x.rsi14 > 50), (x.rsi14 < 50), 24, none, block
    raise ValueError(name)


def simulate(x, pair, name):
    ls, ss, sd, xl, xs, maxb, force, block = signals(x, name)
    ls, ss, sd = np.asarray(ls), np.asarray(ss), np.asarray(sd, float)
    xl, xs, block = np.asarray(xl), np.asarray(xs), np.asarray(block)
    t, o, h, l, c = x.index, x.open.values, x.high.values, x.low.values, x.close.values
    out, pos, pend = [], None, None
    for j in range(len(x)):
        if pend is not None:
            s, a = pend
            pos, pend = (s, o[j], o[j] - s * a, a, j), None
        if pos:
            s, e, stp, a, j0 = pos
            x_ = None
            if j > j0 and ((s > 0 and o[j] <= stp) or (s < 0 and o[j] >= stp)):
                x_ = o[j]
            elif (s > 0 and l[j] <= stp) or (s < 0 and h[j] >= stp):
                x_ = stp
            elif (xl[j] if s > 0 else xs[j]) or j - j0 >= maxb or force[j]:
                x_ = c[j]
            if x_ is None:
                continue
            out.append((t[j0], t[j], s, (x_ - e) * s / a, e, a))
            pos = None
            continue
        if not block[j] and np.isfinite(sd[j]) and sd[j] > 0 and j + 1 < len(x):
            if ls[j]:
                pend = (1, sd[j])
            elif ss[j]:
                pend = (-1, sd[j])
    tr = pd.DataFrame(out, columns=["entry", "exit", "side", "Rbrut", "px", "risk"])
    nights = ((tr.exit - pd.Timedelta(hours=22)).dt.normalize() - (tr.entry - pd.Timedelta(hours=22)).dt.normalize()).dt.days
    sw = np.where(tr.side > 0, SWAP[pair][0], SWAP[pair][1])  # bp/nuit, + = crédit
    tr["R"] = tr.Rbrut - COST[pair] * 1e-4 * tr.px / tr.risk + sw * 1e-4 * tr.px * nights / tr.risk
    tr["pair"] = pair
    return tr


def ftmo(tr, starts, risk, open_risk=3.0, daily_stop=2.0, fail=6.0):
    """Événements entrée/sortie dans l'ordre chronologique ; gains comptés à la clôture des positions."""
    ent, ext, R = tr.entry.values, tr.exit.values, tr.R.values
    ev = np.concatenate([ent, ext])
    kind = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
    idx = np.r_[np.arange(len(tr)), np.arange(len(tr))]
    order = np.lexsort((-kind, ev))  # à heure égale : sorties d'abord
    ev, kind, idx = ev[order], kind[order], idx[order]
    day = pd.DatetimeIndex(ev).tz_localize("UTC").tz_convert("Europe/Prague").normalize().values
    maxopen = int(round(open_risk / risk))

    def phase(k, target):
        eq, d0, cur, open_, dead = 0.0, 0.0, None, set(), False
        while k < len(ev):
            if day[k] != cur:
                cur, d0, dead = day[k], eq, False
            i = idx[k]
            if kind[k] == 0:
                if not dead and len(open_) < maxopen:
                    open_.add(i)
            elif i in open_:
                open_.discard(i)
                eq += R[i] * risk
                if d0 - eq >= daily_stop:
                    dead = True
                if eq <= -fail:
                    return False, k
                if eq >= target:
                    return True, k
            k += 1
        return None, k

    res = []
    for s in starts:
        k = int(np.searchsorted(ev, np.datetime64(s.tz_convert(None))))
        p1, k1 = phase(k, 10.0)
        if p1 is None:
            continue
        if not p1:
            res.append((False, np.nan))
            continue
        p2, k2 = phase(k1 + 1, 5.0)
        if p2 is None:
            continue
        res.append((p2, (pd.Timestamp(ev[min(k2, len(ev) - 1)]) - s.tz_convert(None)).days))
    r = pd.DataFrame(res, columns=["ok", "jours"])
    r["ok"] = r.ok.astype(bool)
    return r.ok.mean() * 100, r[r.ok].jours.median(), len(r)


def tstat(r):
    return r.mean() / r.std() * np.sqrt(len(r))


def main():
    dfold, hfold = sys.argv[1], sys.argv[2]
    pairs = MAJ + CRO
    data = {"J": {p: load(dfold, p, True) for p in pairs}, "H": {p: load(hfold, p, False) for p in pairs}}
    rows, keep = [], {}
    for name in ("J-RSI2", "J-BOLL", "H-ASIE", "H-RSI"):
        tr = pd.concat([simulate(data[name[0]][p], p, name) for p in pairs]).sort_values("entry").reset_index(drop=True)
        keep[name] = tr
        for per, (a, b) in (("IS", IS), ("OOS", OOS)):
            s = tr[(tr.entry >= pd.Timestamp(a, tz="UTC")) & (tr.entry <= pd.Timestamp(b, tz="UTC"))].reset_index(drop=True)
            yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
            bypair = s.groupby("pair").R.mean()
            row = dict(strategie=name, periode=per, trades_an=round(len(s) / yrs), gagnants=round((s.R > 0).mean() * 100),
                       espR_brut=round(s.Rbrut.mean(), 3), espR_net=round(s.R.mean(), 3), t_net=round(tstat(s.R), 1),
                       paires_positives=f"{(bypair > 0).sum()}/{len(bypair)}")
            for risk in (0.5, 1.0):
                pr, dur, n = ftmo(s, pd.date_range(a, b, freq="MS", tz="UTC"), risk)
                row[f"ftmo_{risk}%"] = f"{pr:.0f}% ({n}) {dur:.0f}j"
            rows.append(row)
        print(name, "ok", file=sys.stderr)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    res.to_csv("lab4_results.csv", index=False)
    for name, tr in keep.items():
        y = tr.groupby(tr.entry.dt.year).R.sum().round(0)
        print(f"{name} R net par an:", y.astype(int).to_dict())


if __name__ == "__main__":
    main()
