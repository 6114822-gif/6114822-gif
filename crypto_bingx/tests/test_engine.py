"""Тесты движка на детерминированных сценариях и на случайном блуждании."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_bingx.backtest.costs import CostModel
from crypto_bingx.backtest.metrics import summarize
from crypto_bingx.strategies.crypto_vladimir_engine import CryptoVladimirEngine
from crypto_bingx.strategies.levels import atr_known_at_open, find_pivots, wilder_atr
from crypto_bingx.strategies.params import VladimirParams
from crypto_bingx.tests.synthetic import bars_from_path, random_walk_m5

LEVEL = 1100.0


def _daily_with_level(days: int = 20, pivot_day: int = 10) -> pd.DataFrame:
    """Флэт 950–1050 и один пивот-хай 1100 в день pivot_day."""
    idx = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    high = np.full(days, 1050.0)
    low = np.full(days, 950.0)
    high[pivot_day] = LEVEL
    return pd.DataFrame({"open": 1000.0, "high": high, "low": low, "close": 1000.0,
                         "volume": 1.0}, index=idx)


def _params(**kw) -> VladimirParams:
    base = dict(stop_atr_multiplier=0.3, target_atr_multipliers=(1.0, 1.5, 2.0),
                h1_require_confirmation=False, enable_reaction=False, atr_period=5)
    base.update(kw)
    return VladimirParams(**base)


def _run(path, params, costs=CostModel.zero(), day="2024-01-17"):
    daily = _daily_with_level()
    m5 = bars_from_path(path, day)
    eng = CryptoVladimirEngine(params, costs)
    res = eng.run(m5, "TEST", daily=daily)
    atr = float(atr_known_at_open(daily, params.atr_period).loc[pd.Timestamp(day, tz="UTC")])
    return res, atr


# ------------------------------------------------------------------ уровни

def test_pivot_available_only_after_right_bars():
    daily = _daily_with_level()
    piv = find_pivots(daily, 2, 2)
    assert len(piv) == 1
    row = piv.iloc[0]
    assert row.price == LEVEL and row.kind == "high"
    assert row.pivot_day == daily.index[10]
    assert row.available_from == daily.index[13]   # день 10 + 2 правых бара + 1


def test_atr_known_at_open_is_shifted():
    daily = _daily_with_level()
    atr = wilder_atr(daily, 5)
    known = atr_known_at_open(daily, 5)
    assert known.iloc[7] == atr.iloc[6]


# ------------------------------------------------------------------ сценарии

def _breakout_path(after_fill):
    pre = [(1090, 1091, 1089, 1090)] * 5
    trigger = [(1090, 1096, 1089, 1095), (1095, 1103, 1094, 1102), (1102, 1105, 1101.5, 1104)]
    fill = [(1104, 1104.5, 1100.5, 1103)]
    return pre + trigger + fill + after_fill


def test_breakout_long_all_targets_zero_costs():
    after = [(1103, 1350, 1095, 1340)]   # следующий бар проходит все цели, стоп не задет
    res, atr = _run(_breakout_path(after), _params())
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == 1 and t.setup == "breakout"
    tol = 0.01 * atr
    assert t.entry_price == pytest.approx(LEVEL + tol)
    assert t.stop_price == pytest.approx(t.entry_price - 0.3 * atr)
    assert t.exit_reason == "target" and t.targets_hit == 3
    # (1 + 1.5 + 2) / 3 ATR при риске 0.3 ATR = 5R
    assert t.net_r == pytest.approx(5.0)
    assert t.fee_r == 0 and t.slippage_r == 0


def test_stop_first_when_bar_hits_stop_and_target():
    after = [(1103, 1400, 900, 1000)]    # один бар задевает и стоп, и все цели
    res, _ = _run(_breakout_path(after), _params())
    t = res.trades[0]
    assert t.exit_reason == "stop"
    # гэпа нет (open выше стопа) → ровно −1R
    assert t.net_r == pytest.approx(-1.0)


def test_stop_gap_fills_at_open():
    after = [(1000, 1001, 990, 995)]     # бар открылся сильно ниже стопа
    res, atr = _run(_breakout_path(after), _params())
    t = res.trades[0]
    expected = (1000 - t.entry_price) / (0.3 * atr)
    assert t.net_r == pytest.approx(expected)
    assert t.net_r < -1.0


def test_costs_are_charged_in_r():
    after = [(1103, 1104, 900, 950)]     # стоп
    costs = CostModel(maker_fee=0.0002, taker_fee=0.0005, slippage_limit=0.001,
                      slippage_market=0.001, use_funding=False)
    res, atr = _run(_breakout_path(after), _params(), costs=costs)
    t = res.trades[0]
    risk = 0.3 * atr
    entry_fill = t.entry_price * 1.001
    exit_fill = t.stop_price * 0.999
    assert t.gross_r == pytest.approx(-1.0)
    assert t.fee_r == pytest.approx((0.0002 * entry_fill + 0.0005 * exit_fill) / risk)
    assert t.slippage_r == pytest.approx((entry_fill - t.entry_price + t.stop_price - exit_fill) / risk)
    assert t.net_r == pytest.approx(-1.0 - t.fee_r - t.slippage_r)


def test_limit_not_filled_on_touch_and_cancelled_on_runaway():
    # после сигнала цена касается цены лимитки ровно (без прохода) и уходит к T1
    limit = LEVEL + 0.01 * _atr()
    path = _breakout_path([])[:8] + [(1104, 1104.5, limit, 1103), (1103, 1250, 1102, 1240)]
    res, _ = _run(path, _params())
    assert res.trades == []
    assert res.cancelled_orders == 1


def _atr():
    return float(atr_known_at_open(_daily_with_level(), 5).loc[pd.Timestamp("2024-01-17", tz="UTC")])


def test_funding_charged_for_long_at_funding_time():
    # сделка через 08:00 UTC: лонг при положительной ставке платит
    day = "2024-01-17"
    path = _breakout_path([(1103, 1104, 1090, 1100)] * 120)
    daily = _daily_with_level()
    m5 = bars_from_path(path, day)
    funding = pd.DataFrame({"funding_rate": [0.001]},
                           index=pd.DatetimeIndex([pd.Timestamp(f"{day} 08:00", tz="UTC")]))
    costs = CostModel(maker_fee=0, taker_fee=0, slippage_limit=0, slippage_market=0, use_funding=True)
    res = CryptoVladimirEngine(_params(), costs).run(m5, "T", funding=funding, daily=daily)
    t = res.trades[0]
    assert t.funding_r == pytest.approx(-0.001 * 1103 / (0.3 * _atr()))


# ------------------------------------------------------------------ свойства на случайных данных

def test_no_lookahead_truncation_invariance():
    """Сделки, закрытые до момента T, не должны зависеть от данных после T."""
    m5 = random_walk_m5(days=150, seed=3)
    p = VladimirParams()
    full = CryptoVladimirEngine(p, CostModel.base()).run(m5, "S").trades_frame()
    cut = m5.index[len(m5) * 2 // 3]
    part = CryptoVladimirEngine(p, CostModel.base()).run(m5[m5.index < cut], "S").trades_frame()
    a = full[full.exit_time < cut - pd.Timedelta(days=1)].reset_index(drop=True)
    b = part[part.exit_time < cut - pd.Timedelta(days=1)].reset_index(drop=True)
    assert len(a) > 20
    pd.testing.assert_frame_equal(a, b)


def test_random_walk_has_no_gross_edge():
    """На случайном блуждании без издержек преимущества быть не должно."""
    m5 = random_walk_m5(days=720, seed=11)
    res = CryptoVladimirEngine(VladimirParams(), CostModel.zero()).run(m5, "S")
    s = summarize(res.trades_frame(), res.start, res.end)
    assert s["trades"] > 200
    assert abs(s["avg_r"]) < 0.1


def test_params_validation():
    with pytest.raises(ValueError):
        VladimirParams(target_fractions=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        VladimirParams(target_atr_multipliers=(2.0, 1.0, 3.0))
    p = VladimirParams.scaled_from_forts(2, stop_atr_multiplier=0.45)
    assert p.target_atr_multipliers == (1.0, 1.5, 2.0) and p.stop_atr_multiplier == 0.45
    assert VladimirParams.forts_baseline().stop_atr_multiplier == 0.15


def test_reaction_short_from_resistance():
    # касание снизу → отскок вниз → повторное касание с закрытием под уровнем
    path = ([(1090, 1091, 1089, 1090)] * 5
            + [(1090, 1099, 1089, 1095),     # касание 1
               (1095, 1096, 1087, 1088),     # отскок ≥ 0.08 ATR
               (1088, 1098.5, 1087, 1094),   # касание 2 → сигнал шорт
               (1094, 1099.5, 1093, 1095),   # цена проходит лимитку 1099 → вход
               (1095, 1096, 850, 860)])      # все цели
    res, atr = _run(path, _params(enable_breakout=False, enable_reaction=True))
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == -1 and t.setup == "reaction"
    assert t.entry_price == pytest.approx(LEVEL - 0.01 * atr)
    assert t.stop_price == pytest.approx(t.entry_price + 0.3 * atr)
    assert t.net_r == pytest.approx(5.0)


def test_h1_confirmation_rules():
    eng = CryptoVladimirEngine(VladimirParams())
    atr = 100.0

    def arr(bars):
        a = np.array(bars, dtype=float)
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    # база: 4 H1-бара в диапазоне 30 (≤ 0.35 ATR) у уровня 1100
    base = [(1085, 1095, 1080, 1090)] * 2 + [(1090, 1098, 1082, 1092)] * 2
    assert eng._h1_confirms(1, LEVEL, atr, 3, *arr(base))
    # нож: последний H1 длиной 60 (> 0.5 ATR) — отказ
    knife = base[:3] + [(1040, 1100, 1040, 1098)]
    assert not eng._h1_confirms(1, LEVEL, atr, 3, *arr(knife))
    # далеко от уровня и без реакции — отказ
    far = [(1000, 1010, 995, 1005)] * 4
    assert not eng._h1_confirms(1, LEVEL, atr, 3, *arr(far))
    # реакция: широкий (не база), но без ножа; последний H1 коснулся уровня и закрылся выше
    react = [(1040, 1080, 1040, 1075), (1075, 1095, 1060, 1090), (1090, 1099, 1085, 1098),
             (1098, 1112, 1096, 1108)]
    assert eng._h1_confirms(1, LEVEL, atr, 3, *arr(react))
    assert not eng._h1_confirms(-1, LEVEL, atr, 3, *arr(react))
