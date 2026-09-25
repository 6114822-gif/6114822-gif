"""Ресемплинг OHLCV (M5 → H1 → D1) по UTC.

Сутки на крипте начинаются в 00:00 UTC — так же режут дневную свечу Binance
и BingX. Метка бара — время открытия (label='left', closed='left').
"""
from __future__ import annotations

import pandas as pd

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Собрать старший таймфрейм. Пустые интервалы (нет ни одного бара) выбрасываются.

    rule: '1h', '4h', '1D' и т.п. (синтаксис pandas).
    """
    agg = {k: v for k, v in _AGG.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open"])
    # Сколько младших баров попало в каждый старший — пригодится для контроля неполных баров.
    out["n_bars"] = df["close"].resample(rule, label="left", closed="left").count().reindex(out.index)
    return out
