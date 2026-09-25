"""Шаг 1.2 плана: проверка гипотезы «издержки съедают стоп 0.15 ATR».

Для каждой монеты по реальным дневным данным считает:
  • ATR(D) в % цены — медиана и разброс по годам;
  • размер стопа в % для множителей 0.15 / 0.3 / 0.45 / 0.6 ATR;
  • издержки сделки в R для трех сценариев:
      base-loss   — вход maker, выход стопом (taker) + проскальзывание стопа;
      base-win    — вход maker, выход лимиткой на цели (maker);
      stress      — 0.1 % проскальзывания на вход и выход (из ТЗ);
  • средний |funding| за одно начисление в R.

Запуск (после download_binance_data):
    python -m crypto_bingx.research.cost_vs_stop
    python -m crypto_bingx.research.cost_vs_stop --symbols BTCUSDT ETHUSDT SOLUSDT --out docs/reports/cost_vs_stop.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crypto_bingx import config
from crypto_bingx.backtest.costs import CostModel
from crypto_bingx.data import storage
from crypto_bingx.data.resample import resample_ohlcv
from crypto_bingx.strategies.levels import atr_known_at_open

STOP_MULTS = (0.15, 0.3, 0.45, 0.6)


def load_daily(store: Path, symbol: str) -> pd.DataFrame:
    df = storage.load_frame(storage.klines_path(store, symbol, "1d"))
    if df is None:
        df = resample_ohlcv(storage.load_klines(store, symbol, "5m"), "1D")
    return df


def analyze_symbol(daily: pd.DataFrame, funding: pd.DataFrame | None, atr_period: int = 14,
                   stop_mults=STOP_MULTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (статистика ATR по годам, издержки в R по годам и множителям)."""
    atr_pct = (atr_known_at_open(daily, atr_period) / daily["open"]).dropna()
    by_year = atr_pct.groupby(atr_pct.index.year)
    atr_stats = pd.DataFrame({
        "days": by_year.size(),
        "atr_pct_p10": by_year.quantile(0.10) * 100,
        "atr_pct_median": by_year.median() * 100,
        "atr_pct_p90": by_year.quantile(0.90) * 100,
    })
    atr_stats.loc["all"] = [len(atr_pct), atr_pct.quantile(0.1) * 100, atr_pct.median() * 100,
                            atr_pct.quantile(0.9) * 100]

    base, stress = CostModel.base(), CostModel.stress()
    fund_abs = None
    if funding is not None and len(funding):
        fund_abs = funding["funding_rate"].abs()
    rows = []
    for year, med in list(by_year.median().items()) + [("all", atr_pct.median())]:
        fa = None
        if fund_abs is not None:
            sel = fund_abs if year == "all" else fund_abs[fund_abs.index.year == year]
            fa = sel.mean() if len(sel) else None
        for m in stop_mults:
            stop_pct = m * med
            rows.append({
                "year": year, "stop_atr": m, "stop_pct": stop_pct * 100,
                "cost_r_base_loss": base.round_trip_cost_pct(True, False) / stop_pct,
                "cost_r_base_win": base.round_trip_cost_pct(True, True) / stop_pct,
                "cost_r_stress": stress.round_trip_cost_pct(True, False) / stop_pct,
                "funding_r_per_event": (fa / stop_pct) if fa is not None else None,
            })
    return atr_stats, pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(store: Path, symbols: list[str]) -> str:
    parts = [f"# Издержки против размера стопа (шаг 1.2)\n",
             f"Сгенерировано: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
             f"Модель издержек: base = {CostModel.base()}, stress = {CostModel.stress()}.\n",
             "Как читать: `cost_r_*` — какая доля 1R уходит на издержки одной сделки. "
             "На FORTS это ≈ 0.01–0.02R. Если здесь > 0.2R — преимущество стратегии "
             "с винрейтом ~50 % и средним выигрышем ~2R съедается более чем на треть.\n"]
    for sym in symbols:
        daily = load_daily(store, sym)
        funding = storage.load_funding(store, sym)
        atr_stats, costs = analyze_symbol(daily, funding)
        parts.append(f"\n## {sym}\n\n### ATR(D), % цены\n")
        parts.append(_md_table(atr_stats.reset_index().rename(columns={"index": "year"})))
        parts.append("\n\n### Издержки в R (по медианному ATR года)\n")
        parts.append(_md_table(costs))
        parts.append("\n")
    return "\n".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(config.DEFAULT_SYMBOLS))
    p.add_argument("--store", type=Path, default=config.STORE_DIR)
    p.add_argument("--out", type=Path, default=config.REPORTS_DIR / "cost_vs_stop.md")
    args = p.parse_args(argv)
    report = build_report(args.store, [s.upper() for s in args.symbols])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nСохранено: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
