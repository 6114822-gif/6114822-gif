"""Хранение рыночных данных в Parquet.

Раскладка файлов:
    <store>/binance_um/klines/<interval>/<SYMBOL>.parquet
    <store>/binance_um/funding/<SYMBOL>.parquet

Индекс — время открытия бара (UTC, tz-aware), колонки: open, high, low,
close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume.
Для funding индекс — время начисления, колонка funding_rate (и mark_price,
если биржа его отдала).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SOURCE = "binance_um"


def klines_path(store: Path, symbol: str, interval: str) -> Path:
    return Path(store) / SOURCE / "klines" / interval / f"{symbol.upper()}.parquet"


def funding_path(store: Path, symbol: str) -> Path:
    return Path(store) / SOURCE / "funding" / f"{symbol.upper()}.parquet"


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Атомарная запись: сначала во временный файл, потом rename.

    Если процесс упадет посреди записи, старый файл останется целым.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def load_frame(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Единое разрешение времени — наносекунды (pandas 3 может читать в us/ms).
    df.index = df.index.as_unit("ns")
    return df


def load_klines(store: Path, symbol: str, interval: str) -> pd.DataFrame:
    """Загрузить свечи; бросает понятную ошибку, если данных нет."""
    df = load_frame(klines_path(store, symbol, interval))
    if df is None:
        raise FileNotFoundError(
            f"Нет данных {symbol} {interval} в {store}. "
            f"Сначала запусти: python -m crypto_bingx.data.download_binance_data "
            f"--symbols {symbol} --intervals {interval}"
        )
    return df


def load_funding(store: Path, symbol: str) -> pd.DataFrame | None:
    return load_frame(funding_path(store, symbol))


def merge_frames(old: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame | None:
    """Склеить два куска по индексу: дубли убираются, приоритет у новых данных."""
    frames = [f for f in (old, new) if f is not None and len(f)]
    if not frames:
        return old if old is not None else new
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()
