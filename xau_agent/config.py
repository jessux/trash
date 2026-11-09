"""Règles FTMO 2-step + marges de sécurité internes (plus strictes que FTMO)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FtmoRules:
    initial_balance: float = 10_000.0
    max_daily_loss_pct: float = 5.0      # vs solde du début de journée (minuit Prague)
    max_loss_pct: float = 10.0           # equity jamais < 90% du capital initial
    phase1_target_pct: float = 10.0
    phase2_target_pct: float = 5.0
    min_trading_days: int = 4


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.5      # 0.5% du capital initial par trade
    internal_daily_stop_pct: float = 2.0 # on coupe bien avant les 5% FTMO
    internal_max_loss_pct: float = 6.0   # on coupe bien avant les 10% FTMO
    max_trades_per_day: int = 2
    max_consecutive_losses_day: int = 2
    max_spread_usd: float = 0.50         # pas d'entrée si spread XAUUSD > 0.50$
    contract_size: float = 100.0         # 1 lot XAUUSD = 100 oz
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 2.0


@dataclass(frozen=True)
class StrategyConfig:
    ema_period: int = 200
    donchian_period: int = 20
    atr_period: int = 14
    sl_atr: float = 1.5
    tp_atr: float = 3.0                  # R:R = 2
    session_start_utc: int = 7           # Londres
    session_end_utc: int = 19            # avant fin NY
    friday_flat_utc: int = 19            # aucune position le week-end
