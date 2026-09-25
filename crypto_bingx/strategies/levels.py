"""Дневные уровни и ATR без заглядывания в будущее.

Ключевое правило: всё, что используется внутри дня D, должно быть известно на
момент 00:00 UTC дня D, то есть посчитано по дням строго ДО D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_atr(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR по Уайлдеру (RMA от True Range), значение на закрытии каждого дня."""
    high, low, close = daily["high"], daily["low"], daily["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_known_at_open(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR, известный к началу каждого дня = ATR на закрытии предыдущего дня."""
    return wilder_atr(daily, period).shift(1)


def find_pivots(daily: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Подтвержденные пивоты дневного графика.

    Пивот-хай в день i: high[i] строго выше `left` баров слева и не ниже
    `right` баров справа. Аналогично пивот-лоу. Пивот можно использовать
    только с дня i + right + 1 (после закрытия последнего правого бара).

    Возвращает DataFrame: pivot_day, available_from, price, kind ('high'|'low').
    """
    h = daily["high"].to_numpy()
    lo = daily["low"].to_numpy()
    days = daily.index
    rows = []
    for i in range(left, len(daily) - right):
        if i + right + 1 >= len(daily):
            available = days[i] + pd.Timedelta(days=right + 1)
        else:
            available = days[i + right + 1]
        lh, rh = h[i - left:i], h[i + 1:i + right + 1]
        if h[i] > lh.max() and h[i] >= rh.max():
            rows.append((days[i], available, float(h[i]), "high"))
        ll, rl = lo[i - left:i], lo[i + 1:i + right + 1]
        if lo[i] < ll.min() and lo[i] <= rl.min():
            rows.append((days[i], available, float(lo[i]), "low"))
    return pd.DataFrame(rows, columns=["pivot_day", "available_from", "price", "kind"])


def active_levels(pivots: pd.DataFrame, day: pd.Timestamp, lookback_days: int,
                  merge_distance: float) -> np.ndarray:
    """Уровни, действующие в день `day`: пивоты, подтвержденные до начала дня,
    не старше lookback_days. Близкие уровни (< merge_distance) сливаются —
    остается самый свежий.
    """
    if pivots.empty:
        return np.empty(0)
    mask = (pivots["available_from"] <= day) & (
        pivots["pivot_day"] >= day - pd.Timedelta(days=lookback_days))
    sel = pivots.loc[mask].sort_values("pivot_day", ascending=False)
    kept: list[float] = []
    for price in sel["price"].to_numpy():
        if all(abs(price - k) >= merge_distance for k in kept):
            kept.append(float(price))
    return np.array(sorted(kept))
