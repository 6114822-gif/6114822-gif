"""Прогон CryptoVladimirEngine по сетке параметров и профилям издержек.

Это НЕ оптимизация (она будет walk-forward на этапе 2), а первичная карта:
как результат зависит от ширины стопа и масштаба целей, и сколько съедают
издержки. Для каждой комбинации печатается сводка; сделки сохраняются в
<store>/results/, сводка — в docs/reports/.

Примеры:
    # сетка из ТЗ: стоп 0.3–0.6 ATR, цели FORTS ×2 и ×3, база + стресс, плюс H0
    python -m crypto_bingx.backtest.run_backtest --forts-baseline

    # офлайн-проверка на синтетике (без данных)
    python -m crypto_bingx.backtest.run_backtest --synthetic --symbols SYN --stop-mults 0.3 --target-scales 2
"""
from __future__ import annotations

import argparse
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crypto_bingx import config
from crypto_bingx.backtest.costs import CostModel
from crypto_bingx.backtest.metrics import by_period, summarize
from crypto_bingx.data import storage
from crypto_bingx.strategies.crypto_vladimir_engine import CryptoVladimirEngine
from crypto_bingx.strategies.params import VladimirParams

log = logging.getLogger("backtest")

SUMMARY_COLS = ["symbol", "variant", "costs", "trades", "trades_per_year", "winrate", "profit_factor",
                "avg_r", "r_per_year", "gross_total_r", "avg_cost_r", "cost_share_of_gross",
                "max_dd_r", "max_dd_pct", "cagr_pct", "sharpe", "worst_day_pct", "r_by_year"]


def _load(symbol: str, store: Path, start, end, synthetic: bool):
    if synthetic:
        from crypto_bingx.tests.synthetic import random_walk_m5
        m5 = random_walk_m5(days=365 * 2, seed=sum(map(ord, symbol)))  # стабильный seed (hash() случаен между запусками)
        return m5, None
    m5 = storage.load_klines(store, symbol, "5m")
    funding = storage.load_funding(store, symbol)
    if start is not None:
        m5 = m5[m5.index >= start]
    if end is not None:
        m5 = m5[m5.index < end]
    return m5, funding


def _job(args) -> tuple[dict, pd.DataFrame]:
    symbol, variant, params, costs, store, start, end, synthetic, risk = args
    m5, funding = _load(symbol, store, start, end, synthetic)
    res = CryptoVladimirEngine(params, costs).run(m5, symbol, funding=funding)
    trades = res.trades_frame()
    s = summarize(trades, res.start, res.end, risk_per_trade=risk)
    s.update({"symbol": symbol, "variant": variant, "costs": costs.name})
    if len(trades):
        yr = by_period(trades, "YE")["total_r"]
        s["r_by_year"] = " ".join(f"{p}:{v:+.1f}" for p, v in yr.items())
    trades.insert(0, "variant", variant)
    trades.insert(1, "costs", costs.name)
    return s, trades


def build_variants(stop_mults, target_scales, forts_baseline: bool) -> list[tuple[str, VladimirParams]]:
    out = []
    if forts_baseline:
        out.append(("H0 forts (stop 0.15, T×1)", VladimirParams.forts_baseline()))
    for stop, scale in itertools.product(stop_mults, target_scales):
        out.append((f"stop {stop} ATR, T×{scale}",
                    VladimirParams.scaled_from_forts(scale, stop_atr_multiplier=stop)))
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return "" if v is None else str(v)


def to_markdown(summary: pd.DataFrame) -> str:
    cols = [c for c in SUMMARY_COLS if c in summary.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summary[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(config.DEFAULT_SYMBOLS))
    p.add_argument("--stop-mults", nargs="+", type=float, default=[0.3, 0.45, 0.6])
    p.add_argument("--target-scales", nargs="+", type=float, default=[2.0, 3.0])
    p.add_argument("--costs", nargs="+", default=["base", "stress"], choices=["zero", "base", "stress"])
    p.add_argument("--forts-baseline", action="store_true", help="добавить H0: параметры FORTS как есть")
    p.add_argument("--start", type=lambda s: pd.Timestamp(s, tz="UTC"))
    p.add_argument("--end", type=lambda s: pd.Timestamp(s, tz="UTC"))
    p.add_argument("--risk", type=float, default=0.01, help="риск на сделку для equity-метрик (0.01 = 1 %%)")
    p.add_argument("--store", type=Path, default=config.STORE_DIR)
    p.add_argument("--out", type=Path, default=None, help="markdown-отчет (по умолчанию docs/reports/)")
    p.add_argument("--jobs", type=int, default=1, help="параллельных процессов")
    p.add_argument("--synthetic", action="store_true", help="случайное блуждание вместо данных (проверка)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    variants = build_variants(args.stop_mults, args.target_scales, args.forts_baseline)
    jobs = [(sym.upper(), name, params, CostModel.by_name(c), args.store, args.start, args.end,
             args.synthetic, args.risk)
            for sym in args.symbols for name, params in variants for c in args.costs]
    log.info("Прогонов: %d", len(jobs))
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]

    summary = pd.DataFrame([r[0] for r in results])
    trades = pd.concat([r[1] for r in results], ignore_index=True) if results else pd.DataFrame()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    tag = "synthetic" if args.synthetic else "binance"
    md = (f"# Прогон сетки CryptoVladimir ({tag})\n\n"
          f"Сгенерировано: {stamp} UTC. Период: {args.start or 'вся история'} — {args.end or 'конец данных'}. "
          f"Риск на сделку для equity-метрик: {args.risk:.2%}.\n\n"
          f"⚠️ Это первичная карта параметров, не оптимизация и не OOS-результат.\n\n"
          + to_markdown(summary) + "\n")
    out = args.out or (config.REPORTS_DIR / f"grid_{tag}_{stamp}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    res_dir = Path(args.store) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(res_dir / f"trades_{tag}_{stamp}.csv", index=False)
    summary.to_csv(res_dir / f"summary_{tag}_{stamp}.csv", index=False)
    print(md)
    print(f"Отчет: {out}\nСделки: {res_dir / f'trades_{tag}_{stamp}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
