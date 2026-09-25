"""Модель торговых издержек BingX USDT-M perpetual.

Все значения — доли (0.0005 = 0.05 %). Проскальзывание всегда работает против
нас. Отдельно для лимитных (maker) и рыночных/стоп (taker) исполнений.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    maker_fee: float = 0.0002        # лимитный вход, тейк-профиты
    taker_fee: float = 0.0005        # стоп-маркет, выход по времени
    slippage_limit: float = 0.0      # «проскальзывание» лимитных ордеров (стресс-надбавка)
    slippage_market: float = 0.0002  # проскальзывание рыночных исполнений
    use_funding: bool = True         # учитывать исторический funding rate
    name: str = "base"

    @classmethod
    def base(cls) -> "CostModel":
        """Реалистичная оценка для BTC/ETH при небольшом размере позиции."""
        return cls()

    @classmethod
    def stress(cls) -> "CostModel":
        """Стресс-сценарий из ТЗ: 0.1 % на вход и 0.1 % на выход даже для лимитных ордеров."""
        return cls(slippage_limit=0.001, slippage_market=0.001, name="stress")

    @classmethod
    def zero(cls) -> "CostModel":
        """Без издержек — чтобы увидеть валовое преимущество сигнала."""
        return cls(maker_fee=0.0, taker_fee=0.0, slippage_limit=0.0, slippage_market=0.0,
                   use_funding=False, name="zero")

    @classmethod
    def by_name(cls, name: str) -> "CostModel":
        presets = {"base": cls.base, "stress": cls.stress, "zero": cls.zero}
        if name not in presets:
            raise ValueError(f"Неизвестный профиль издержек '{name}', есть: {sorted(presets)}")
        return presets[name]()

    def fee(self, is_maker: bool) -> float:
        return self.maker_fee if is_maker else self.taker_fee

    def slippage(self, is_maker: bool) -> float:
        return self.slippage_limit if is_maker else self.slippage_market

    def round_trip_cost_pct(self, entry_maker: bool = True, exit_maker: bool = False) -> float:
        """Издержки на круг в долях цены (без funding)."""
        return (self.fee(entry_maker) + self.slippage(entry_maker)
                + self.fee(exit_maker) + self.slippage(exit_maker))
