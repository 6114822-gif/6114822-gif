"""Проверка качества свечей: пропуски, дубли, битые OHLC, выбросы.

Результат — dataclass, который сериализуется в JSON-отчет загрузчика.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_bingx.data.resample import resample_ohlcv

_FREQ = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
         "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D"}


@dataclass
class QualityReport:
    interval: str
    bars: int = 0
    start: str | None = None
    end: str | None = None
    expected_bars: int = 0
    missing_bars: int = 0
    missing_pct: float = 0.0
    duplicates: int = 0
    ohlc_violations: int = 0          # high < max(open, close) или low > min(open, close) и т.п.
    non_positive_prices: int = 0
    zero_volume_bars: int = 0
    outlier_bars: int = 0             # диапазон бара > outlier_k × скользящей медианы диапазона
    largest_gaps: list = field(default_factory=list)   # [(начало, конец, пропущено баров)]

    def summary(self) -> str:
        return (f"{self.bars} баров {self.start} → {self.end}; пропущено {self.missing_bars} "
                f"({self.missing_pct:.3f}%), дублей {self.duplicates}, битых OHLC {self.ohlc_violations}, "
                f"нулевой объем {self.zero_volume_bars}, выбросов {self.outlier_bars}")


def check_klines(df: pd.DataFrame, interval: str, outlier_k: float = 15.0,
                 top_gaps: int = 10) -> QualityReport:
    rep = QualityReport(interval=interval)
    if df is None or df.empty:
        return rep
    freq = pd.Timedelta(_FREQ[interval])
    rep.bars = int(len(df))
    rep.start, rep.end = str(df.index[0]), str(df.index[-1])
    rep.duplicates = int(df.index.duplicated().sum())
    idx = df.index[~df.index.duplicated()]
    rep.expected_bars = int((idx[-1] - idx[0]) / freq) + 1
    rep.missing_bars = max(rep.expected_bars - len(idx), 0)
    rep.missing_pct = 100.0 * rep.missing_bars / max(rep.expected_bars, 1)

    # Разрывы: шаг между соседними барами больше одного интервала.
    deltas = idx[1:] - idx[:-1]
    gap_mask = deltas > freq
    if gap_mask.any():
        gaps = pd.DataFrame({"start": idx[:-1][gap_mask], "end": idx[1:][gap_mask],
                             "missing": (deltas[gap_mask] / freq).astype(int) - 1})
        gaps = gaps.sort_values("missing", ascending=False).head(top_gaps)
        rep.largest_gaps = [(str(r.start), str(r.end), int(r.missing)) for r in gaps.itertuples()]

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    rep.ohlc_violations = int(np.sum((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)))
    rep.non_positive_prices = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    if "volume" in df:
        rep.zero_volume_bars = int((df["volume"] <= 0).sum())

    rng = pd.Series(h - l, index=df.index)
    med = rng.rolling(288, min_periods=20).median()
    rep.outlier_bars = int((rng > outlier_k * med).sum())
    return rep


def compare_daily(m5: pd.DataFrame, d1: pd.DataFrame) -> dict:
    """Сверить D1, собранный из M5, с официальной дневной свечой.

    Большое расхождение = в M5 есть пропуски или ошибки. Сравниваем только дни,
    полностью покрытые M5 (288 баров).
    """
    agg = resample_ohlcv(m5, "1D")
    agg = agg[agg["n_bars"] == 288]
    common = agg.index.intersection(d1.index)
    if len(common) == 0:
        return {"days_compared": 0}
    a, b = agg.loc[common], d1.loc[common]
    out = {"days_compared": int(len(common))}
    for col in ("open", "high", "low", "close"):
        rel = (a[col] - b[col]).abs() / b[col]
        out[f"{col}_max_rel_diff"] = float(rel.max())
        out[f"{col}_days_diff_gt_1bp"] = int((rel > 1e-4).sum())
    return out
