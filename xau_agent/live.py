"""Agent Exécution MT5 (Windows + terminal MetaTrader 5 FTMO ouvert). DRY-RUN par défaut.

Usage: python live.py            -> simule, n'envoie aucun ordre
       python live.py --live     -> envoie les ordres (uniquement après backtest + démo validés)
"""
import argparse
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from config import FtmoRules, RiskConfig, StrategyConfig
from risk import RiskManager
from strategy import indicators, signal

SYMBOL, MAGIC = "XAUUSD", 777001
logging.basicConfig(filename="journal.log", level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def main():
    import MetaTrader5 as mt5  # pip install MetaTrader5 (Windows uniquement)

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--target", type=float, default=FtmoRules.phase1_target_pct)
    a = ap.parse_args()
    scfg = StrategyConfig()
    rm = RiskManager(FtmoRules(), RiskConfig(), a.target)
    if not mt5.initialize():
        raise SystemExit(f"MT5 init échoué: {mt5.last_error()}")
    log.info("start live=%s target=%s", a.live, a.target)
    last_bar = None

    def close_all():
        for p in mt5.positions_get(symbol=SYMBOL) or []:
            if p.magic != MAGIC:
                continue
            tick = mt5.symbol_info_tick(SYMBOL)
            buy = p.type == mt5.POSITION_TYPE_SELL
            req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYMBOL, volume=p.volume, position=p.ticket,
                       type=mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
                       price=tick.ask if buy else tick.bid, deviation=20, magic=MAGIC)
            log.info("CLOSE %s", mt5.order_send(req) if a.live else req)

    while True:
        acc = mt5.account_info()
        now = datetime.now(timezone.utc)
        prague_day = pd.Timestamp(now).tz_convert("Europe/Prague").date()
        rm.new_day(prague_day, acc.balance)
        positions = [p for p in (mt5.positions_get(symbol=SYMBOL) or []) if p.magic == MAGIC]

        reason = rm.check(acc.balance, acc.equity)
        friday = now.weekday() == 4 and now.hour >= scfg.friday_flat_utc
        if (reason or friday) and positions:
            log.info("coupe-circuit: %s", reason or "vendredi")
            close_all()
        if rm.halted:
            log.info("ARRET: %s", rm.halted)
            break

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 1, 400)  # bougies clôturées
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = indicators(df.set_index("time"), scfg)
        ts = df.index[-1]
        if ts != last_bar:
            last_bar = ts
            tick = mt5.symbol_info_tick(SYMBOL)
            spread = tick.ask - tick.bid
            sig = signal(df.iloc[-1], ts, scfg)
            block = rm.can_open(acc.balance, acc.equity, spread, bool(positions))
            if sig and not block:
                price = tick.ask if sig.side > 0 else tick.bid
                eurusd = mt5.symbol_info_tick("EURUSD").bid
                lots = rm.lot_size(abs(price - sig.sl), eurusd, acc.equity)
                if lots > 0:
                    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=SYMBOL, volume=lots, price=price,
                               type=mt5.ORDER_TYPE_BUY if sig.side > 0 else mt5.ORDER_TYPE_SELL,
                               sl=round(sig.sl, 2), tp=round(sig.tp, 2), deviation=20, magic=MAGIC,
                               type_filling=mt5.ORDER_FILLING_IOC, comment="xau_agent")
                    res = mt5.order_send(req) if a.live else "DRY-RUN"
                    log.info("OPEN %s -> %s", req, res)
                    rm.on_open(prague_day)
            elif sig:
                log.info("signal bloqué: %s", block)
        time.sleep(5)


if __name__ == "__main__":
    main()
