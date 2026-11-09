"""Backtest des 2 phases FTMO sur un CSV OHLC XAUUSD (colonnes: time,open,high,low,close ; heure UTC).

Usage: python backtest.py data/XAUUSD_M15.csv [--eurusd 1.08] [--spread 0.30]
Hypothèses pessimistes : si SL et TP touchés dans la même bougie -> SL ; equity évaluée au pire de la bougie.
"""
import argparse
from dataclasses import dataclass, field

import pandas as pd

from config import FtmoRules, RiskConfig, StrategyConfig
from risk import RiskManager
from strategy import indicators, signal


@dataclass
class Result:
    phase: str
    passed: bool
    reason: str
    start: pd.Timestamp
    end: pd.Timestamp
    final_balance: float
    trades: list = field(default_factory=list)
    ftmo_violation: str | None = None
    trading_days: int = 0

    def summary(self) -> str:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t > 0)
        return (f"[{self.phase}] {'REUSSI' if self.passed else 'ECHEC'} - {self.reason} | "
                f"{self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} | solde {self.final_balance:.0f} | "
                f"trades {n} | winrate {wins / n * 100 if n else 0:.0f}% | jours {self.trading_days}"
                + (f" | VIOLATION FTMO: {self.ftmo_violation}" if self.ftmo_violation else ""))


def run_phase(df, start_i, target_pct, phase, eurusd, spread, rules=FtmoRules(),
              rcfg=RiskConfig(), scfg=StrategyConfig()) -> tuple[Result, int]:
    rm = RiskManager(rules, rcfg, target_pct)
    balance = rules.initial_balance
    pos = None  # (side, entry, sl, tp, lots)
    pnls = []
    usd_to_eur = lambda usd: usd / eurusd
    violation = None
    idx = df.index
    i = start_i
    for i in range(start_i, len(df)):
        ts, row = idx[i], df.iloc[i]
        d = ts.tz_convert("Europe/Prague").date()
        if rm.day != d:
            rm.new_day(d, balance)
            ftmo_day_floor = balance - rules.initial_balance * rules.max_daily_loss_pct / 100

        # gestion position ouverte
        if pos:
            side, entry, sl, tp, lots = pos
            k = lots * rcfg.contract_size
            worst = row["low"] if side > 0 else row["high"]
            equity_worst = balance + usd_to_eur((worst - entry) * side * k)
            if equity_worst < ftmo_day_floor and not violation:
                violation = f"daily loss {ts}"
            if equity_worst < rules.initial_balance * (1 - rules.max_loss_pct / 100) and not violation:
                violation = f"max loss {ts}"
            hit_sl = (row["low"] <= sl) if side > 0 else (row["high"] >= sl)
            hit_tp = (row["high"] >= tp) if side > 0 else (row["low"] <= tp)
            friday_flat = ts.weekday() == 4 and ts.hour >= scfg.friday_flat_utc
            exit_px = sl if hit_sl else tp if hit_tp else row["close"] if friday_flat else None
            if exit_px is None and rm.check(balance, equity_worst):
                exit_px = row["close"]  # coupe-circuit interne
            if exit_px is not None:
                pnl = usd_to_eur((exit_px - entry) * side * k)
                balance += pnl
                pnls.append(pnl)
                rm.on_close(pnl)
                pos = None

        equity = balance
        stop = rm.check(balance, equity)
        if rm.halted:
            break
        if stop or pos:
            continue
        sig = signal(row, ts, scfg)
        if not sig or rm.can_open(balance, equity, spread, False):
            continue
        entry = sig.entry + spread / 2 * sig.side  # paie le demi-spread
        lots = rm.lot_size(abs(entry - sig.sl), eurusd, equity)
        if lots > 0:
            pos = (sig.side, entry, sig.sl, sig.tp, lots)
            rm.on_open(d)

    passed = (rm.halted or "").startswith("OBJECTIF") and len(rm.trading_days) >= rules.min_trading_days \
        and violation is None
    reason = rm.halted or "fin des données"
    return Result(phase, passed, reason, idx[start_i], idx[i], balance, pnls, violation,
                  len(rm.trading_days)), i + 1


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--eurusd", type=float, default=1.08)
    ap.add_argument("--spread", type=float, default=0.30)
    a = ap.parse_args()
    df = indicators(load(a.csv), StrategyConfig())
    r1, nxt = run_phase(df, 0, FtmoRules.phase1_target_pct, "Phase 1", a.eurusd, a.spread)
    print(r1.summary())
    if r1.passed:
        r2, nxt = run_phase(df, nxt, FtmoRules.phase2_target_pct, "Phase 2", a.eurusd, a.spread)
        print(r2.summary())
        if r2.passed:
            r3, _ = run_phase(df, nxt, 100.0, "Funded x2", a.eurusd, a.spread)
            print(r3.summary())


if __name__ == "__main__":
    main()
