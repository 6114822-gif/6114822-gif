"""Тесты скриптов исследования на синтетических данных."""
from __future__ import annotations

import pandas as pd
import pytest

from crypto_bingx.backtest.costs import CostModel
from crypto_bingx.data import storage
from crypto_bingx.data.resample import resample_ohlcv
from crypto_bingx.research import cost_vs_stop
from crypto_bingx.tests.synthetic import random_walk_m5


def test_cost_vs_stop_report(tmp_path):
    m5 = random_walk_m5(days=400, start="2023-06-01")
    storage.save_frame(m5, storage.klines_path(tmp_path, "SYNUSDT", "5m"))
    idx = pd.date_range("2023-06-01", "2024-07-01", freq="8h", tz="UTC")
    storage.save_frame(pd.DataFrame({"funding_rate": 0.0001}, index=idx),
                       storage.funding_path(tmp_path, "SYNUSDT"))
    out = tmp_path / "r.md"
    assert cost_vs_stop.main(["--symbols", "SYNUSDT", "--store", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## SYNUSDT" in text and "cost_r_stress" in text

    daily = resample_ohlcv(m5, "1D")
    atr_stats, costs = cost_vs_stop.analyze_symbol(daily, None)
    row = costs[(costs.year == "all") & (costs.stop_atr == 0.3)].iloc[0]
    med = atr_stats.loc["all", "atr_pct_median"] / 100
    assert row.cost_r_base_loss == pytest.approx(CostModel.base().round_trip_cost_pct(True, False) / (0.3 * med))
    # чем шире стоп, тем меньше издержки в R
    allrows = costs[costs.year == "all"].sort_values("stop_atr")
    assert allrows.cost_r_base_loss.is_monotonic_decreasing
