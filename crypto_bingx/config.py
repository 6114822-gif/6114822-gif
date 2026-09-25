"""Общая конфигурация проекта crypto_bingx.

Здесь только пути и значения по умолчанию. Параметры стратегии живут в
strategies/params.py, модель издержек — в backtest/costs.py.
"""
from __future__ import annotations

import os
from pathlib import Path

# Корень пакета crypto_bingx/
PROJECT_ROOT = Path(__file__).resolve().parent

# Куда складываются скачанные данные (Parquet). Папка в .gitignore.
# Можно переопределить переменной окружения CRYPTO_BINGX_STORE.
STORE_DIR = Path(os.environ.get("CRYPTO_BINGX_STORE", PROJECT_ROOT / "data" / "store"))

# Куда пишутся отчеты исследований (коммитятся в git).
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"

# Стартовый набор инструментов для этапа 1 (USDT-M бессрочные фьючерсы).
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

# Таймфреймы, которые качаем. H1 собирается из M5 ресемплингом (точно и без
# расхождений), D1 качаем отдельно — для сверки с ресемплингом M5 и для ATR.
DEFAULT_INTERVALS = ("5m", "1d")

# Глубина истории по умолчанию, лет.
DEFAULT_YEARS = 3
