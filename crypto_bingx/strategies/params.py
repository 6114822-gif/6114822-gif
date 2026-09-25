"""Параметры стратегии CryptoVladimir.

Все величины цен выражены в долях дневного ATR (как в Vladimir Core на FORTS),
поэтому одни и те же параметры применимы к BTC за 60 000 $ и к SOL за 150 $.

Базовая линия FORTS: стоп 0.15 ATR, цели 0.5 / 0.75 / 1.0 ATR.
Для крипты стоп расширяем (0.3–0.6 ATR), цели масштабируем (×2, ×3), —
см. docs/00_PLAN.md, раздел 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

FORTS_STOP_ATR = 0.15
FORTS_TARGETS_ATR = (0.5, 0.75, 1.0)


@dataclass(frozen=True)
class VladimirParams:
    # --- Дневные уровни (D) ---
    atr_period: int = 14                  # период дневного ATR (Wilder)
    pivot_left: int = 2                   # баров слева у подтвержденного пивота
    pivot_right: int = 2                  # баров справа (пивот доступен только после их закрытия)
    level_lookback_days: int = 90         # сколько дней назад берем пивоты
    level_merge_atr: float = 0.1          # уровни ближе этого (в ATR) сливаются в один

    # --- Риск и цели (в долях ATR(D)) ---
    stop_atr_multiplier: float = 0.3                       # стоп от цены входа
    target_atr_multipliers: tuple = (1.0, 1.5, 2.0)        # T1, T2, T3 (FORTS × 2)
    target_fractions: tuple = (1 / 3, 1 / 3, 1 / 3)       # доля позиции на каждую цель
    breakeven_after_t1: bool = False                       # перевод стопа в безубыток после T1

    # --- Вход ---
    entry_tolerance_atr: float = 0.01     # допуск лимитного ордера от уровня («2–3 тика» на FORTS)
    order_ttl_bars: int = 12              # сколько M5-баров ждем исполнения лимитки (12 = 1 час)
    max_trades_per_day: int = 2           # максимум входов на инструмент за UTC-сутки
    max_hold_bars: int = 288              # выход по времени (288 × M5 = 24 часа)
    trade_hours_utc: tuple | None = None  # (с, до) — окно открытия сделок по UTC; None = 24/7

    # --- Сетапы ---
    enable_breakout: bool = True
    enable_reaction: bool = True

    # --- M5-триггер ---
    breakout_closes: int = 2              # пробой: минимум N закрытий M5 за уровнем
    touch_tolerance_atr: float = 0.03     # что считаем «касанием» уровня
    bounce_min_atr: float = 0.08          # минимальный отскок между двумя касаниями
    reaction_max_bars: int = 48           # касание→отскок→касание должно уложиться в N баров M5
    level_break_atr: float = 0.03         # закрытие за уровнем дальше этого = уровень сломан (сброс реакции)

    # --- H1-подтверждение ---
    h1_knife_lookback: int = 3            # сколько последних H1 проверяем на «нож/палку»
    h1_knife_max_atr: float = 0.5         # H1-бар с диапазоном больше этого = импульс без базы
    h1_base_bars: int = 4                 # длина базы (сжатия) в H1-барах
    h1_base_max_atr: float = 0.35         # диапазон базы не больше этого
    h1_base_proximity_atr: float = 0.15   # база должна быть у уровня (не дальше этого)
    h1_require_confirmation: bool = True  # False — отключить H1-фильтр (для сравнения)

    notes: str = field(default="", compare=False)

    def __post_init__(self):
        if len(self.target_atr_multipliers) != len(self.target_fractions):
            raise ValueError("Число целей и долей позиции должно совпадать")
        if abs(sum(self.target_fractions) - 1.0) > 1e-9:
            raise ValueError("Сумма долей target_fractions должна быть 1.0")
        if self.stop_atr_multiplier <= 0:
            raise ValueError("stop_atr_multiplier должен быть > 0")
        if list(self.target_atr_multipliers) != sorted(self.target_atr_multipliers):
            raise ValueError("Цели должны идти по возрастанию")

    @classmethod
    def forts_baseline(cls, **overrides) -> "VladimirParams":
        """H0: параметры FORTS как есть (стоп 0.15 ATR, цели 0.5/0.75/1.0 ATR)."""
        base = cls(stop_atr_multiplier=FORTS_STOP_ATR, target_atr_multipliers=FORTS_TARGETS_ATR,
                   notes="H0: FORTS как есть")
        return replace(base, **overrides)

    @classmethod
    def scaled_from_forts(cls, scale: float, stop_atr_multiplier: float | None = None,
                          **overrides) -> "VladimirParams":
        """Цели FORTS × scale. Стоп — тоже × scale, если не задан явно."""
        stop = stop_atr_multiplier if stop_atr_multiplier is not None else FORTS_STOP_ATR * scale
        targets = tuple(round(t * scale, 6) for t in FORTS_TARGETS_ATR)
        base = cls(stop_atr_multiplier=stop, target_atr_multipliers=targets,
                   notes=f"FORTS цели ×{scale}, стоп {stop} ATR")
        return replace(base, **overrides)

    def with_(self, **overrides) -> "VladimirParams":
        return replace(self, **overrides)

    def to_dict(self) -> dict:
        return asdict(self)
