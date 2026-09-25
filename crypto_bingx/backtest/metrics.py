"""Метрики бэктеста: в R (как в Vladimir Core) и в % депозита при фиксированном риске.

Equity-кривая строится упрощенно: каждая сделка меняет капитал на
risk_per_trade × net_R от капитала на момент закрытия (сделки одного
инструмента не пересекаются по времени). Портфельная симуляция с общими
лимитами — в модуле риска (этап 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, start: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None, risk_per_trade: float = 0.01) -> dict:
    """Сводка по сделкам. trades — BacktestResult.trades_frame()."""
    n = int(len(trades))
    out: dict = {"trades": n}
    if n == 0:
        return out
    r = trades["net_r"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    years = None
    if start is not None and end is not None:
        years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)
    gross_profit, gross_loss = wins.sum(), -losses.sum()
    out.update({
        "winrate": float(len(wins) / n),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_r": float(r.mean()),
        "total_r": float(r.sum()),
        "r_per_year": float(r.sum() / years) if years else None,
        "trades_per_year": float(n / years) if years else None,
        "gross_total_r": float(trades["gross_r"].sum()),
        "fees_total_r": float(trades["fee_r"].sum()),
        "slippage_total_r": float(trades["slippage_r"].sum()),
        "funding_total_r": float(trades["funding_r"].sum()),
        "avg_cost_r": float((trades["fee_r"] + trades["slippage_r"] - trades["funding_r"]).mean()),
        "max_dd_r": float(max_drawdown_r(r.to_numpy())),
        "max_losing_streak": int(max_losing_streak(r.to_numpy())),
    })
    gross = out["gross_total_r"]
    costs = out["fees_total_r"] + out["slippage_total_r"] - out["funding_total_r"]
    out["cost_share_of_gross"] = float(costs / gross) if gross > 0 else None
    out.update(equity_stats(trades, start, end, risk_per_trade))
    return out


def max_drawdown_r(r: np.ndarray) -> float:
    if len(r) == 0:
        return 0.0
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def max_losing_streak(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def equity_curve(trades: pd.DataFrame, risk_per_trade: float = 0.01,
                 initial: float = 1.0) -> pd.Series:
    """Капитал после каждой сделки (по времени выхода), фиксированный % риска."""
    if trades.empty:
        return pd.Series([initial], dtype=float)
    t = trades.sort_values("exit_time")
    growth = 1.0 + risk_per_trade * t["net_r"].astype(float).to_numpy()
    eq = initial * np.cumprod(growth)
    return pd.Series(eq, index=pd.DatetimeIndex(t["exit_time"]))


def equity_stats(trades: pd.DataFrame, start, end, risk_per_trade: float) -> dict:
    eq = equity_curve(trades, risk_per_trade)
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    out = {"risk_per_trade": risk_per_trade, "final_equity_x": float(eq.iloc[-1]),
           "max_dd_pct": float(-min(dd, 0.0) * 100)}
    if start is None or end is None:
        return out
    # Дневные доходности: дни без закрытых сделок = 0 (крипта торгуется 7 дней в неделю).
    days = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    daily_eq = eq.groupby(eq.index.floor("D")).last().reindex(days).ffill().fillna(1.0)
    rets = daily_eq.pct_change().fillna(daily_eq.iloc[0] - 1.0)
    std = rets.std()
    years = max(len(days) / 365.25, 1e-9)
    out["cagr_pct"] = float((eq.iloc[-1] ** (1 / years) - 1) * 100)
    out["sharpe"] = float(rets.mean() / std * np.sqrt(365)) if std > 0 else None
    worst_day = rets.min()
    out["worst_day_pct"] = float(worst_day * 100)
    out["calmar"] = (out["cagr_pct"] / out["max_dd_pct"]) if out["max_dd_pct"] > 0 else None
    return out


def by_period(trades: pd.DataFrame, freq: str = "YE") -> pd.DataFrame:
    """Разбивка результата по годам/кварталам (freq pandas: 'YE', 'QE', 'ME')."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["period"] = pd.DatetimeIndex(t["exit_time"]).tz_convert("UTC").to_period(freq.rstrip("E"))
    g = t.groupby("period")["net_r"]
    out = pd.DataFrame({
        "trades": g.size(),
        "winrate": g.apply(lambda s: (s > 0).mean()),
        "total_r": g.sum(),
        "pf": g.apply(lambda s: s[s > 0].sum() / -s[s <= 0].sum() if (s <= 0).any() and s[s <= 0].sum() < 0 else np.inf),
    })
    return out
