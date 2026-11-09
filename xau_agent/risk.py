"""Agent Risque : seul juge du droit d'ouvrir un trade et de sa taille. Non contournable."""
import math
from dataclasses import dataclass, field
from datetime import date

from config import FtmoRules, RiskConfig


@dataclass
class RiskManager:
    rules: FtmoRules = field(default_factory=FtmoRules)
    cfg: RiskConfig = field(default_factory=RiskConfig)
    target_pct: float = 10.0
    day: date | None = None
    day_start_balance: float = 0.0
    trades_today: int = 0
    losses_in_row_today: int = 0
    trading_days: set = field(default_factory=set)
    halted: str | None = None            # raison d'arrêt définitif

    @property
    def init(self) -> float:
        return self.rules.initial_balance

    def new_day(self, d: date, balance: float) -> None:
        if d != self.day:
            self.day, self.day_start_balance = d, balance
            self.trades_today = self.losses_in_row_today = 0

    def check(self, balance: float, equity: float) -> str | None:
        """Retourne une raison de blocage (et ferme tout côté appelant) ou None."""
        if self.halted:
            return self.halted
        if equity <= self.init * (1 - self.cfg.internal_max_loss_pct / 100):
            self.halted = "STOP DEFINITIF: perte max interne atteinte"
            return self.halted
        if equity >= self.init * (1 + self.target_pct / 100):
            self.halted = "OBJECTIF ATTEINT: on ne trade plus"
            return self.halted
        if self.day_start_balance - equity >= self.init * self.cfg.internal_daily_stop_pct / 100:
            return "stop journalier interne"
        return None

    def can_open(self, balance: float, equity: float, spread: float, has_position: bool) -> str | None:
        reason = self.check(balance, equity)
        if reason:
            return reason
        if has_position:
            return "position deja ouverte"
        if self.trades_today >= self.cfg.max_trades_per_day:
            return "max trades/jour"
        if self.losses_in_row_today >= self.cfg.max_consecutive_losses_day:
            return "pertes consecutives du jour"
        if spread > self.cfg.max_spread_usd:
            return "spread trop large"
        return None

    def lot_size(self, stop_usd: float, eurusd: float, equity: float) -> float:
        """Taille telle que perte au SL = risk_per_trade (en EUR), bornée par la marge restante."""
        if stop_usd <= 0:
            return 0.0
        risk_eur = self.init * self.cfg.risk_per_trade_pct / 100
        # ne jamais risquer plus que ce qui reste avant le stop journalier / max interne
        room_day = self.init * self.cfg.internal_daily_stop_pct / 100 - (self.day_start_balance - equity)
        room_max = equity - self.init * (1 - self.cfg.internal_max_loss_pct / 100)
        risk_eur = min(risk_eur, room_day, room_max)
        if risk_eur <= 0:
            return 0.0
        lots = risk_eur * eurusd / (stop_usd * self.cfg.contract_size)
        lots = math.floor(lots / self.cfg.lot_step) * self.cfg.lot_step
        lots = min(round(lots, 2), self.cfg.max_lot)
        return lots if lots >= self.cfg.min_lot else 0.0

    def on_open(self, d: date) -> None:
        self.trades_today += 1
        self.trading_days.add(d)

    def on_close(self, pnl: float) -> None:
        self.losses_in_row_today = self.losses_in_row_today + 1 if pnl < 0 else 0
