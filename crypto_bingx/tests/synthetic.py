"""Синтетические данные для офлайн-тестов (без доступа к бирже)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_m5(days: int = 120, start: str = "2023-01-01", price: float = 30_000.0,
                   daily_vol: float = 0.03, seed: int = 7) -> pd.DataFrame:
    """Случайное блуждание M5 с реалистичной дневной волатильностью (~3 %)."""
    rng = np.random.default_rng(seed)
    n = days * 288
    step_vol = daily_vol / np.sqrt(288)
    rets = rng.standard_t(df=4, size=n) * step_vol / np.sqrt(2)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[price], close[:-1]])
    wick = np.abs(rng.normal(0, step_vol * 0.6, size=(2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    vol = rng.uniform(50, 150, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


def bars_from_path(path: list[tuple[float, float, float, float]], start: str) -> pd.DataFrame:
    """Собрать M5 из явного списка (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(path), freq="5min", tz="UTC")
    arr = np.array(path, dtype=float)
    return pd.DataFrame({"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
                         "close": arr[:, 3], "volume": 1.0}, index=idx)
