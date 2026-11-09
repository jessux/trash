"""Suivi de tendance diversifié (27 marchés FTMO) en données journalières, simulé comme un compte FTMO.

Système canonique, paramètres NON optimisés (règles "Turtle" publiques) :
- entrée : clôture > plus haut 55 j (achat) ou < plus bas 55 j (vente), exécution à l'ouverture suivante ;
- stop initial 2 x ATR(20) (ordre stop, gap exécuté à l'ouverture) ; sortie : clôture sous le plus bas 20 j
  (achat) / au-dessus du plus haut 20 j (vente), exécution à l'ouverture suivante ;
- variante de robustesse : 20 / 10 jours.
Risque : r % du capital initial par position, risque ouvert total <= 4 %, max 3 positions par secteur.
Coûts (prudents) : spread+commission en bp à l'entrée, swap chaque nuit quel que soit le sens.
Données : OANDA journalier (huggingface.co/datasets/bunqi/OANDA_*), fichiers <SYM>_<décennie>.parquet.
Usage: python lab3.py <dossier>
"""
import glob
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

#            secteur,   coût bp, swap bp/nuit
MARKETS = {
    "SPX500USD": ("indices", 2, 2), "NAS100USD": ("indices", 2, 2), "US2000USD": ("indices", 3, 2),
    "DE30EUR": ("indices", 2, 2), "UK100GBP": ("indices", 2, 2), "EU50EUR": ("indices", 2, 2),
    "JP225USD": ("indices", 3, 2), "AU200AUD": ("indices", 3, 2), "HK33HKD": ("indices", 4, 2),
    "EURUSD": ("devises", 1, 0.5), "GBPUSD": ("devises", 1, 0.5), "USDJPY": ("devises", 1, 0.5),
    "AUDUSD": ("devises", 1, 0.5), "USDCAD": ("devises", 1, 0.5), "USDCHF": ("devises", 1, 0.5),
    "NZDUSD": ("devises", 1.5, 0.5),
    "XAUUSD": ("metaux", 1.5, 1.5), "XAGUSD": ("metaux", 4, 1.5), "XPTUSD": ("metaux", 10, 1.5),
    "XPDUSD": ("metaux", 10, 1.5),
    "WTICOUSD": ("energie", 3, 2), "BCOUSD": ("energie", 3, 2), "NATGASUSD": ("energie", 10, 2),
    "CORNUSD": ("agricole", 10, 2), "WHEATUSD": ("agricole", 10, 2), "SOYBNUSD": ("agricole", 10, 2),
    "SUGARUSD": ("agricole", 10, 2),
}
SECTOR_MAX, OPEN_RISK_MAX = 3, 4.0


def load(folder, entry=55, exit_=20):
    data = {}
    for sym in MARKETS:
        x = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{folder}/{sym}_*.parquet")))
        # bougie OANDA ouverte à 17h New York = séance du jour suivant
        x.index = (pd.to_datetime(x.pop("time"), unit="s") + pd.Timedelta(hours=7)).dt.normalize()
        x = x[~x.index.duplicated(keep="last")].sort_index()
        tr = pd.concat([x.high - x.low, (x.high - x.close.shift()).abs(), (x.low - x.close.shift()).abs()], axis=1).max(axis=1)
        x["atr"] = tr.rolling(20).mean()
        x["hi_e"], x["lo_e"] = x.high.rolling(entry).max().shift(1), x.low.rolling(entry).min().shift(1)
        x["hi_x"], x["lo_x"] = x.high.rolling(exit_).max().shift(1), x.low.rolling(exit_).min().shift(1)
        x["gap_days"] = x.index.to_series().diff().dt.days.fillna(1)
        data[sym] = x[["open", "high", "low", "close", "atr", "hi_e", "lo_e", "hi_x", "lo_x", "gap_days"]]
    cal = sorted(set().union(*(d.index for d in data.values())))
    # matrices alignées sur le calendrier commun (NaN = pas de séance)
    return pd.DatetimeIndex(cal), {k: v.reindex(cal).to_numpy() for k, v in data.items()}


O, H, L, C, ATR, HIE, LOE, HIX, LOX, GAP = range(10)


def run(cal, M, t0, risk=0.5, target=None, fail=None, t_end=None, journal=None, swap_mult=1.0):
    """Simule le portefeuille à partir de l'indice t0. Equity en % du capital initial (non composée).
    Retourne (statut, indice de fin, série d'equity, violation FTMO)."""
    syms = list(M)
    pos = {}                       # sym -> [side, entry, stop, stop_dist, exit_pending]
    pend = {}                      # sym -> side (entrée à l'ouverture suivante)
    closed, eq_hist, violation = 0.0, [], None
    day_start = 0.0
    t_end = len(cal) if t_end is None else t_end
    for t in range(t0, t_end):
        for s in syms:
            b = M[s][t]
            if np.isnan(b[O]):
                continue
            _, cost_bp, swap_bp = MARKETS[s]
            p = pos.get(s)
            if p:
                side, e, stp, sd, ex, cr = p
                sw = swap_mult * swap_bp * 1e-4 * b[O] * b[GAP] / sd
                p[5] += sw
                cr += sw
                closed -= risk * sw
                x = None
                if ex:
                    x = b[O]
                elif (side > 0 and b[O] <= stp) or (side < 0 and b[O] >= stp):
                    x = b[O]
                elif (side > 0 and b[L] <= stp) or (side < 0 and b[H] >= stp):
                    x = stp
                if x is not None:
                    r = (x - e) * side / sd
                    closed += risk * r
                    if journal is not None:
                        journal.append((s, cal[t], r, r - cr))
                    del pos[s]
            if s in pend and s not in pos:
                side = pend.pop(s)
                sec = MARKETS[s][0]
                n_sec = sum(1 for k in pos if MARKETS[k][0] == sec)
                if n_sec < SECTOR_MAX and (len(pos) + 1) * risk <= OPEN_RISK_MAX and not np.isnan(M[s][t - 1][ATR]):
                    sd = 2 * M[s][t - 1][ATR]
                    e = b[O]
                    closed -= risk * cost_bp * 1e-4 * e / sd
                    stp = e - side * sd
                    pos[s] = [side, e, stp, sd, False, cost_bp * 1e-4 * e / sd]
                    if (side > 0 and b[L] <= stp) or (side < 0 and b[H] >= stp):   # stop touché le jour même
                        closed -= risk
                        if journal is not None:
                            journal.append((s, cal[t], -1.0, -1.0 - pos[s][5]))
                        del pos[s]
            pend.pop(s, None)
            # signaux à la clôture
            if s in pos:
                side = pos[s][0]
                if (side > 0 and b[C] < b[LOX]) or (side < 0 and b[C] > b[HIX]):
                    pos[s][4] = True
            elif b[C] > b[HIE]:
                pend[s] = 1
            elif b[C] < b[LOE]:
                pend[s] = -1
        # valorisation
        flo = worst = 0.0
        for s, (side, e, stp, sd, _, _) in pos.items():
            b = M[s][t]
            last = b if not np.isnan(b[C]) else M[s][t - 1]
            flo += risk * (last[C] - e) * side / sd
            w = last[L] if side > 0 else last[H]
            worst += risk * (w - e) * side / sd
        eq = closed + flo
        eq_worst = closed + worst
        if violation is None and fail is not None:
            if eq_worst <= day_start - 5:
                violation = "perte journalière 5 %"
            elif eq_worst <= -10:
                violation = "perte max 10 %"
        eq_hist.append(eq)
        day_start = eq
        if fail is not None and (eq <= -fail or violation):
            return "echec", t, eq_hist, violation
        if target is not None and eq >= target:
            return "reussi", t, eq_hist, violation
    return "fin", t_end - 1, eq_hist, violation


def challenge(cal, M, starts, risk, swap_mult=1.0):
    res = []
    for t0 in starts:
        s1, t1, _, v1 = run(cal, M, t0, risk, target=10, fail=6, swap_mult=swap_mult)
        if s1 == "fin":
            continue
        if s1 == "echec":
            res.append((False, (cal[t1] - cal[t0]).days, v1))
            continue
        s2, t2, _, v2 = run(cal, M, t1 + 1, risk, target=5, fail=6, swap_mult=swap_mult)
        if s2 == "fin":
            continue
        res.append((s2 == "reussi", (cal[t2] - cal[t0]).days, v2))
    r = pd.DataFrame(res, columns=["ok", "jours", "violation"])
    return r


def main():
    folder = sys.argv[1]
    pd.set_option("display.width", 200)
    for entry, exit_ in ((55, 20), (20, 10)):
        cal, M = load(folder, entry, exit_)
        t0 = int(np.searchsorted(cal, pd.Timestamp("2006-06-01")))
        print(f"\n===== Système {entry}/{exit_} jours, risque 0,5 %/position (2006-06 → 2026-11) =====")
        for sm in (1.0, 0.5, 0.0):
            journal = []
            _, _, eq, _ = run(cal, M, t0, 0.5, journal=journal, swap_mult=sm)
            eq = pd.Series(eq, index=cal[t0:])
            ye = eq.groupby(eq.index.year).last()
            yearly = ye.diff().fillna(ye.iloc[0])
            j = pd.DataFrame(journal, columns=["sym", "date", "R", "Rnet"])
            print(f"\n-- swap x{sm}: {len(j)} trades ({len(j) / 20.4:.0f}/an) | esp. brute {j.R.mean():.3f} R | nette {j.Rnet.mean():.3f} R"
                  f" | gagnants {(j.R > 0).mean() * 100:.0f} % | total {eq.iloc[-1]:.0f} % | max DD {(eq.cummax() - eq).max():.1f} %"
                  f" | années positives {(yearly > 0).sum()}/{len(yearly)}")
            print("   par an:", {k: round(v, 1) for k, v in yearly.items()})
            print("   par secteur (R net):", j.assign(sec=j.sym.map(lambda x: MARKETS[x][0])).groupby("sec").Rnet.mean().round(2).to_dict())
            for label, a, b in (("2006-2015", "2006-06-01", "2015-12-31"), ("2016-2026", "2016-01-01", "2026-11-05")):
                starts = [int(np.searchsorted(cal, d)) for d in pd.date_range(a, b, freq="MS")]
                for risk in (0.5, 1.0):
                    r = challenge(cal, M, starts, risk, sm)
                    ok = r[r.ok]
                    print(f"   FTMO {label} risque {risk}%: réussite {r.ok.mean() * 100:.0f}% ({len(r)} départs) | "
                          f"durée médiane {ok.jours.median():.0f} j | violations {r.violation.notna().sum()}")


if __name__ == "__main__":
    main()
