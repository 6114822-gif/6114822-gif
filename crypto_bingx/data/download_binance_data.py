"""Загрузка исторических данных Binance USDT-M Futures (свечи + funding).

Почему Binance, а не BingX: у Binance самая длинная и чистая публичная история
бессрочных фьючерсов, плюс официальный архив с контрольными суммами. BingX
торгуется в паритете с Binance через арбитраж; расхождение цен проверим
отдельно на общем периоде (этап 1.1 плана).

Источники:
  1. Архив https://data.binance.vision — помесячные ZIP с CSV и файлом .CHECKSUM
     (SHA-256). Быстро, без лимитов API. Используется для всех завершенных месяцев.
  2. REST https://fapi.binance.com — для текущего (незавершенного) месяца, для
     месяцев, которые еще не выложены в архив, и для истории funding rate.
     Авторизация не нужна.

Докачка: если файл уже есть, загрузка продолжается с последнего бара.

Примеры:
    # 3 года M5 и D1 для BTC/ETH/SOL + funding (значения по умолчанию)
    python -m crypto_bingx.data.download_binance_data

    # с явной датой начала (рекомендую 2021-01-01, чтобы захватить бычий 2021 и медвежий 2022)
    python -m crypto_bingx.data.download_binance_data --start 2021-01-01

    # только REST (если домен data.binance.vision недоступен)
    python -m crypto_bingx.data.download_binance_data --source rest
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

from crypto_bingx import config
from crypto_bingx.data import storage
from crypto_bingx.data.quality import check_klines, compare_daily

log = logging.getLogger("download")

ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"
FAPI_BASE = "https://fapi.binance.com"

# Порядок колонок в CSV архива и в ответе /fapi/v1/klines
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
}

REST_KLINES_LIMIT = 1500      # максимум свечей за запрос /fapi/v1/klines
REST_FUNDING_LIMIT = 1000     # максимум записей за запрос /fapi/v1/fundingRate
REST_PAUSE_SEC = 0.3          # пауза между запросами: запрос с limit=1500 весит 10, лимит 2400/мин


# --------------------------------------------------------------------------- HTTP

class HttpClient:
    """Тонкая обертка над requests с повторами.

    Повторяет запрос при сетевых ошибках, 429/418 (лимиты Binance) и 5xx с
    экспоненциальной паузой. 404 — не ошибка: возвращает None (например, архив
    за месяц еще не выложен или монета тогда не торговалась).
    """

    def __init__(self, session: requests.Session | None = None, max_retries: int = 5,
                 backoff_sec: float = 2.0, timeout_sec: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.timeout_sec = timeout_sec
        self.sleep = sleep

    def get(self, url: str, params: dict | None = None) -> requests.Response | None:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_sec)
            except requests.exceptions.ProxyError as exc:
                # Прокси отказал в CONNECT (облачная среда, домен не разрешен) — повторять бессмысленно.
                raise PermissionError(f"Прокси запретил доступ к {url}: домен не открыт в Network access") from exc
            except requests.RequestException as exc:
                last_error = exc
                wait = self.backoff_sec * 2 ** attempt
                log.warning("Сетевая ошибка %s (%s), повтор через %.0f с", url, exc, wait)
                self.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else self.backoff_sec * 2 ** attempt
                log.warning("HTTP %s от %s, повтор через %.0f с", resp.status_code, url, wait)
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                self.sleep(wait)
                continue
            if resp.status_code in (403, 451):
                raise PermissionError(
                    f"HTTP {resp.status_code} от {url}. 403 в облачной среде — домен не открыт в Network access; "
                    f"451 — Binance блокирует регион сервера (например, США)."
                )
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Не удалось получить {url} за {self.max_retries} попыток") from last_error


# --------------------------------------------------------------------------- Парсинг

def parse_klines_csv(raw: bytes) -> pd.DataFrame:
    """CSV из архива → DataFrame. Старые файлы без заголовка, новые — с заголовком."""
    text = raw.decode("utf-8")
    first = text.split("\n", 1)[0].split(",", 1)[0].strip()
    skip = 0 if first.isdigit() else 1
    df = pd.read_csv(io.StringIO(text), header=None, names=KLINE_COLUMNS, skiprows=skip)
    return normalize_klines(df)


def klines_from_rest(rows: list[list]) -> pd.DataFrame:
    """Ответ /fapi/v1/klines (список списков) → DataFrame."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS[: len(rows[0])] if rows else KLINE_COLUMNS)
    return normalize_klines(df)


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Привести типы, индекс = время открытия в UTC, убрать дубли."""
    if df.empty:
        return _empty_klines()
    out = pd.DataFrame(index=df.index)
    open_time = pd.to_numeric(df["open_time"]).astype("int64")
    # Страховка: с 2025 г. часть архивов Binance (спот) перешла на микросекунды.
    if open_time.max() > 10**14:
        open_time = open_time // 1000
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")
    out.index = pd.to_datetime(open_time.values, unit="ms", utc=True).as_unit("ns")
    out.index.name = "open_time"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[["open", "high", "low", "close", "volume", "quote_volume", "trades",
                "taker_buy_volume", "taker_buy_quote_volume"]]


def _empty_klines() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume"]
    idx = pd.DatetimeIndex([], tz="UTC", name="open_time").as_unit("ns")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}, index=idx)


def funding_from_rest(rows: list[dict]) -> pd.DataFrame:
    """Ответ /fapi/v1/fundingRate → DataFrame(funding_rate, mark_price).

    Время начисления у Binance иногда смещено на несколько мс от ровной
    отметки (08:00:00.003) — округляем до секунды.
    """
    if not rows:
        idx = pd.DatetimeIndex([], tz="UTC", name="funding_time")
        return pd.DataFrame({"funding_rate": pd.Series(dtype="float64"),
                             "mark_price": pd.Series(dtype="float64")}, index=idx)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True).dt.floor("s")
    out = pd.DataFrame({
        "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce").values,
        "mark_price": pd.to_numeric(df.get("markPrice", pd.Series([None] * len(df))),
                                    errors="coerce").values,
    }, index=pd.DatetimeIndex(idx, name="funding_time").as_unit("ns"))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """CSV funding из архива: calc_time,funding_interval_hours,last_funding_rate."""
    text = raw.decode("utf-8")
    first = text.split("\n", 1)[0].split(",", 1)[0].strip()
    names = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    df = pd.read_csv(io.StringIO(text), header=None, names=names, skiprows=0 if first.isdigit() else 1)
    rows = [{"fundingTime": int(t), "fundingRate": r}
            for t, r in zip(df["calc_time"], df["last_funding_rate"])]
    return funding_from_rest(rows)


# --------------------------------------------------------------------------- Архив

def archive_month_url(symbol: str, interval: str, year: int, month: int) -> str:
    name = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    return f"{ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{name}"


def archive_day_url(symbol: str, interval: str, day: date) -> str:
    name = f"{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    return f"{ARCHIVE_BASE}/daily/klines/{symbol}/{interval}/{name}"


def archive_funding_month_url(symbol: str, year: int, month: int) -> str:
    name = f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
    return f"{ARCHIVE_BASE}/monthly/fundingRate/{symbol}/{name}"


def _fetch_zip_csv(client: HttpClient, url: str, verify: bool) -> bytes | None:
    """Скачать ZIP из архива, проверить SHA-256, вернуть содержимое первого CSV."""
    resp = client.get(url)
    if resp is None:
        return None
    payload = resp.content
    if verify:
        chk = client.get(url + ".CHECKSUM")
        if chk is not None:
            expected = chk.text.strip().split()[0].lower()
            actual = hashlib.sha256(payload).hexdigest()
            if expected != actual:
                raise ValueError(f"Контрольная сумма не совпала: {url}")
        else:
            log.warning("Нет файла .CHECKSUM для %s — проверка пропущена", url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"В архиве нет CSV: {url}")
        return zf.read(csv_names[0])


def fetch_archive_month(client: HttpClient, symbol: str, interval: str, year: int,
                        month: int, verify: bool = True) -> pd.DataFrame | None:
    """Скачать месяц свечей из архива. None — если архива за этот месяц нет."""
    raw = _fetch_zip_csv(client, archive_month_url(symbol, interval, year, month), verify)
    return parse_klines_csv(raw) if raw is not None else None


def fetch_archive_days(client: HttpClient, symbol: str, interval: str, start: datetime,
                       end: datetime, verify: bool = True) -> pd.DataFrame:
    """Свечи [start, end) из дневных архивов (только полностью завершенные сутки).

    Запасной путь, когда REST fapi недоступен (гео-блокировка): архив
    выкладывается с задержкой ~1 сутки, поэтому последние часы будут пропущены.
    """
    frames = []
    day = start.date()
    while datetime(day.year, day.month, day.day, tzinfo=timezone.utc) < end:
        raw = _fetch_zip_csv(client, archive_day_url(symbol, interval, day), verify)
        if raw is not None:
            frames.append(parse_klines_csv(raw))
        day += timedelta(days=1)
    if not frames:
        return _empty_klines()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


def fetch_archive_funding(client: HttpClient, symbol: str, start: datetime, end: datetime,
                          verify: bool = True) -> pd.DataFrame:
    """Funding из помесячного архива (запасной путь при недоступном REST)."""
    frames = []
    for m_start in _month_starts(start, end):
        raw = _fetch_zip_csv(client, archive_funding_month_url(symbol, m_start.year, m_start.month), verify)
        if raw is not None:
            frames.append(parse_funding_csv(raw))
    if not frames:
        return funding_from_rest([])
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


# --------------------------------------------------------------------------- REST

def fetch_rest_klines(client: HttpClient, symbol: str, interval: str, start_ms: int,
                      end_ms: int, now_ms: int | None = None) -> pd.DataFrame:
    """Свечи [start_ms, end_ms) через /fapi/v1/klines постранично.

    Незакрытая текущая свеча отбрасывается (иначе в данных окажется бар,
    который потом изменится).
    """
    step = INTERVAL_MS[interval]
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    chunks: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = client.get(f"{FAPI_BASE}/fapi/v1/klines", params={
            "symbol": symbol, "interval": interval, "startTime": cursor,
            "endTime": end_ms - 1, "limit": REST_KLINES_LIMIT,
        })
        rows = resp.json() if resp is not None else []
        if not rows:
            break
        chunk = klines_from_rest(rows)
        chunks.append(chunk)
        last_open = int(rows[-1][0])
        next_cursor = last_open + step
        if next_cursor <= cursor:   # защита от зацикливания
            break
        cursor = next_cursor
        if len(rows) < REST_KLINES_LIMIT:
            break
        client.sleep(REST_PAUSE_SEC)
    if not chunks:
        return _empty_klines()
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + step
    df = df[(close_ms <= now_ms)]
    start_ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    return df[(df.index >= start_ts) & (df.index < end_ts)]


def fetch_rest_funding(client: HttpClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """История funding rate через /fapi/v1/fundingRate."""
    frames: list[pd.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = client.get(f"{FAPI_BASE}/fapi/v1/fundingRate", params={
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": REST_FUNDING_LIMIT,
        })
        rows = resp.json() if resp is not None else []
        if not rows:
            break
        frames.append(funding_from_rest(rows))
        last = int(rows[-1]["fundingTime"])
        if last + 1 <= cursor:
            break
        cursor = last + 1
        if len(rows) < REST_FUNDING_LIMIT:
            break
        client.sleep(REST_PAUSE_SEC)
    if not frames:
        return funding_from_rest([])
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- Оркестрация

def _month_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur < end:
        yield cur
        cur = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=timezone.utc)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def download_klines(client: HttpClient, store: Path, symbol: str, interval: str,
                    start: datetime, end: datetime | None = None, source: str = "auto",
                    verify: bool = True) -> pd.DataFrame:
    """Скачать (или докачать) свечи и сохранить в Parquet. Возвращает полный набор.

    source: 'auto' — архив для завершенных месяцев, REST для остального;
            'archive' — только архив; 'rest' — только REST.
    """
    end = end or datetime.now(timezone.utc)
    step = INTERVAL_MS[interval]
    path = storage.klines_path(store, symbol, interval)
    existing = storage.load_frame(path)
    cursor = start
    if existing is not None and len(existing):
        # Докачка: продолжаем со следующего бара после последнего сохраненного.
        last = existing.index[-1].to_pydatetime()
        cursor = max(start, last + timedelta(milliseconds=step))
        log.info("%s %s: есть данные до %s, докачиваю", symbol, interval, existing.index[-1])
    if cursor >= end:
        return existing

    current_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    pieces: list[pd.DataFrame] = []
    for m_start in _month_starts(cursor, end):
        m_end = datetime(m_start.year + (m_start.month == 12), m_start.month % 12 + 1, 1,
                         tzinfo=timezone.utc)
        seg_start, seg_end = max(cursor, m_start), min(m_end, end)
        whole_month = seg_start == m_start and m_start < current_month
        df = None
        if source in ("auto", "archive") and whole_month:
            df = fetch_archive_month(client, symbol, interval, m_start.year, m_start.month, verify)
            if df is not None:
                log.info("%s %s %s: архив, %d баров", symbol, interval, m_start.strftime("%Y-%m"), len(df))
        if df is None and source in ("auto", "rest"):
            try:
                df = fetch_rest_klines(client, symbol, interval, _to_ms(seg_start), _to_ms(seg_end))
                log.info("%s %s %s: REST, %d баров", symbol, interval, m_start.strftime("%Y-%m"), len(df))
            except PermissionError as exc:
                if source != "auto":
                    raise
                log.warning("REST недоступен (%s) — беру дневные архивы", exc)
                df = fetch_archive_days(client, symbol, interval, seg_start, seg_end, verify)
                log.info("%s %s %s: дневной архив, %d баров", symbol, interval, m_start.strftime("%Y-%m"), len(df))
        if df is not None and len(df):
            pieces.append(df[(df.index >= seg_start) & (df.index < seg_end)])
        # Промежуточное сохранение раз в 6 месяцев — чтобы обрыв не стоил всей загрузки.
        if len(pieces) >= 6:
            existing = storage.merge_frames(existing, pd.concat(pieces))
            storage.save_frame(existing, path)
            pieces = []
    if pieces:
        existing = storage.merge_frames(existing, pd.concat(pieces))
    if existing is None:
        existing = _empty_klines()
    storage.save_frame(existing, path)
    return existing


def download_funding(client: HttpClient, store: Path, symbol: str, start: datetime,
                     end: datetime | None = None, verify: bool = True) -> pd.DataFrame:
    end = end or datetime.now(timezone.utc)
    path = storage.funding_path(store, symbol)
    existing = storage.load_frame(path)
    cursor = start
    if existing is not None and len(existing):
        cursor = max(start, existing.index[-1].to_pydatetime() + timedelta(seconds=1))
    try:
        new = fetch_rest_funding(client, symbol, _to_ms(cursor), _to_ms(end))
    except PermissionError as exc:
        log.warning("REST funding недоступен (%s) — беру месячный архив", exc)
        new = fetch_archive_funding(client, symbol, cursor, end, verify)
    merged = storage.merge_frames(existing, new)
    if merged is None:
        merged = new
    storage.save_frame(merged, path)
    log.info("%s funding: %d записей (новых %d)", symbol, len(merged), len(new))
    return merged


def run(symbols: list[str], intervals: list[str], start: datetime, store: Path,
        source: str = "auto", funding: bool = True, verify: bool = True,
        client: HttpClient | None = None) -> dict:
    """Скачать всё и собрать отчет о качестве. Возвращает словарь отчета."""
    client = client or HttpClient()
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                    "start": start.isoformat(), "symbols": {}}
    for symbol in symbols:
        sym_report: dict = {}
        frames: dict[str, pd.DataFrame] = {}
        for interval in intervals:
            df = download_klines(client, store, symbol, interval, start, source=source, verify=verify)
            frames[interval] = df
            q = check_klines(df, interval)
            sym_report[interval] = asdict(q)
            log.info("%s %s: %s", symbol, interval, q.summary())
        if "5m" in frames and "1d" in frames and len(frames["5m"]) and len(frames["1d"]):
            sym_report["m5_vs_d1"] = compare_daily(frames["5m"], frames["1d"])
        if funding:
            f = download_funding(client, store, symbol, start, verify=verify)
            sym_report["funding"] = {
                "records": int(len(f)),
                "first": str(f.index[0]) if len(f) else None,
                "last": str(f.index[-1]) if len(f) else None,
                "mean_rate": float(f["funding_rate"].mean()) if len(f) else None,
            }
        report["symbols"][symbol] = sym_report
    out = Path(store) / storage.SOURCE / "quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Отчет о качестве данных: %s", out)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Загрузка данных Binance USDT-M Futures")
    p.add_argument("--symbols", nargs="+", default=list(config.DEFAULT_SYMBOLS))
    p.add_argument("--intervals", nargs="+", default=list(config.DEFAULT_INTERVALS),
                   choices=sorted(INTERVAL_MS))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--years", type=float, default=config.DEFAULT_YEARS,
                   help="глубина истории в годах (по умолчанию %(default)s)")
    g.add_argument("--start", type=date.fromisoformat, help="дата начала YYYY-MM-DD")
    p.add_argument("--store", type=Path, default=config.STORE_DIR)
    p.add_argument("--source", choices=["auto", "archive", "rest"], default="auto")
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="не проверять SHA-256 архивов")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.start:
        start = datetime(args.start.year, args.start.month, args.start.day, tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - timedelta(days=round(365.25 * args.years))
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        run([s.upper() for s in args.symbols], args.intervals, start, args.store,
            source=args.source, funding=not args.no_funding, verify=not args.no_verify)
    except PermissionError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
