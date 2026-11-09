"""Agent Stratégie : signaux déterministes (testables). Tendance EMA + cassure Donchian + stop ATR."""
from dataclasses import dataclass

import pandas as pd

from config import StrategyConfig


@dataclass
class Signal:
    side: int          # +1 long, -1 short
    entry: float
    sl: float
    tp: float


def indicators(df: pd.DataFrame, c: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = df["close"].ewm(span=c.ema_period, adjust=False).mean()
    df["dc_hi"] = df["high"].rolling(c.donchian_period).max().shift(1)
    df["dc_lo"] = df["low"].rolling(c.donchian_period).min().shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(c.atr_period).mean()
    return df


def signal(row: pd.Series, ts: pd.Timestamp, c: StrategyConfig) -> Signal | None:
    """Évalue la bougie CLÔTURÉE `row` ; entrée au prix de clôture."""
    if pd.isna(row["atr"]) or pd.isna(row["dc_hi"]):
        return None
    if not (c.session_start_utc <= ts.hour < c.session_end_utc):
        return None
    if ts.weekday() == 4 and ts.hour >= c.friday_flat_utc - 2:
        return None
    px, atr = row["close"], row["atr"]
    if px > row["ema"] and px > row["dc_hi"]:
        return Signal(+1, px, px - c.sl_atr * atr, px + c.tp_atr * atr)
    if px < row["ema"] and px < row["dc_lo"]:
        return Signal(-1, px, px + c.sl_atr * atr, px - c.tp_atr * atr)
    return None
