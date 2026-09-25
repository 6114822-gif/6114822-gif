"""Тесты загрузчика без сети: HTTP подменяется фейковым клиентом,
который отдает ответы в форматах архива и REST Binance."""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_bingx.data import download_binance_data as dl
from crypto_bingx.data import storage
from crypto_bingx.data.quality import check_klines, compare_daily

STEP = dl.INTERVAL_MS["5m"]


def _row(open_ms: int, price: float = 100.0) -> list:
    return [open_ms, price, price + 1, price - 1, price + 0.5, 10.0, open_ms + STEP - 1,
            1000.0, 5, 4.0, 400.0, 0]


def _csv(rows: list[list], header: bool) -> bytes:
    lines = []
    if header:
        lines.append(",".join(dl.KLINE_COLUMNS))
    lines += [",".join(str(x) for x in r) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def _zip(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes = b"", json_data=None):
        self.content = content
        self.text = content.decode("utf-8", errors="replace") if content else ""
        self._json = json_data

    def json(self):
        return self._json


class FakeClient(dl.HttpClient):
    """Архив: словарь url → bytes. REST: генерирует бары по запрошенному диапазону."""

    def __init__(self, archive: dict[str, bytes], rest_until_ms: int, rest_from_ms: int = 0,
                 bad_checksum: bool = False):
        super().__init__(sleep=lambda s: None)
        self.archive = archive
        self.rest_until_ms = rest_until_ms
        self.rest_from_ms = rest_from_ms
        self.bad_checksum = bad_checksum
        self.calls: list[str] = []

    def get(self, url, params=None):
        self.calls.append(url)
        if url.startswith(dl.ARCHIVE_BASE):
            if url.endswith(".CHECKSUM"):
                data = self.archive.get(url[: -len(".CHECKSUM")])
                if data is None:
                    return None
                digest = "0" * 64 if self.bad_checksum else hashlib.sha256(data).hexdigest()
                return FakeResponse(f"{digest}  file.zip".encode())
            data = self.archive.get(url)
            return FakeResponse(data) if data is not None else None
        if url.endswith("/fapi/v1/klines"):
            start = max(params["startTime"], self.rest_from_ms)
            start = ((start + STEP - 1) // STEP) * STEP
            end = min(params["endTime"], self.rest_until_ms)
            rows = [_row(t) for t in range(start, end + 1, STEP)][: params["limit"]]
            return FakeResponse(json_data=rows)
        if url.endswith("/fapi/v1/fundingRate"):
            eight_h = 8 * 3600 * 1000
            start = ((params["startTime"] + eight_h - 1) // eight_h) * eight_h
            rows = [{"symbol": params["symbol"], "fundingTime": t + 3, "fundingRate": "0.0001",
                     "markPrice": "100.0"}
                    for t in range(start, min(params["endTime"], self.rest_until_ms), eight_h)]
            return FakeResponse(json_data=rows[: params["limit"]])
        raise AssertionError(f"неожиданный URL {url}")


def _month_rows(year: int, month: int) -> list[list]:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return [_row(int(t.value // 1_000_000)) for t in pd.date_range(start, end, freq="5min", inclusive="left")]


def test_parse_csv_with_and_without_header():
    rows = [_row(1_700_000_000_000), _row(1_700_000_000_000 + STEP)]
    a = dl.parse_klines_csv(_csv(rows, header=False))
    b = dl.parse_klines_csv(_csv(rows, header=True))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 and a.index.tz is not None
    assert a.index[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")


def test_parse_microsecond_timestamps():
    rows = [_row(1_700_000_000_000)]
    rows[0][0] *= 1000
    df = dl.parse_klines_csv(_csv(rows, header=False))
    assert df.index[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")


def test_archive_checksum_mismatch_raises():
    url = dl.archive_month_url("BTCUSDT", "5m", 2024, 1)
    client = FakeClient({url: _zip("x.csv", _csv(_month_rows(2024, 1), True))}, 0, bad_checksum=True)
    with pytest.raises(ValueError, match="Контрольная сумма"):
        dl.fetch_archive_month(client, "BTCUSDT", "5m", 2024, 1)


def test_download_archive_plus_rest_tail_and_resume(tmp_path):
    archive = {}
    for m in (1, 2):
        url = dl.archive_month_url("BTCUSDT", "5m", 2024, m)
        archive[url] = _zip(f"BTCUSDT-5m-2024-0{m}.csv", _csv(_month_rows(2024, m), header=(m == 2)))
    end = datetime(2024, 3, 10, 12, 0, tzinfo=timezone.utc)
    end_ms = int(end.timestamp() * 1000)
    client = FakeClient(archive, rest_until_ms=end_ms, rest_from_ms=int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp() * 1000))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    df = dl.download_klines(client, tmp_path, "BTCUSDT", "5m", start, end=end)
    expected = pd.date_range(start, end, freq="5min", inclusive="left", tz="UTC")
    assert len(df) == len(expected)
    assert (df.index == expected).all()
    assert any("monthly/klines" in c for c in client.calls)
    q = check_klines(df, "5m")
    assert q.missing_bars == 0 and q.duplicates == 0 and q.ohlc_violations == 0

    # докачка: новые бары после end
    end2 = datetime(2024, 3, 11, tzinfo=timezone.utc)
    client2 = FakeClient({}, rest_until_ms=int(end2.timestamp() * 1000))
    df2 = dl.download_klines(client2, tmp_path, "BTCUSDT", "5m", start, end=end2)
    assert df2.index[-1] == pd.Timestamp(end2) - pd.Timedelta("5min")
    assert not any("monthly/klines" in c for c in client2.calls)   # архив повторно не качаем
    assert storage.load_klines(tmp_path, "BTCUSDT", "5m").equals(df2)


def test_rest_drops_unclosed_bar():
    base = 1_699_999_800_000                           # кратно 5 минутам
    now_ms = base + 10 * STEP + 60_000                 # 11-й бар еще не закрыт
    client = FakeClient({}, rest_until_ms=now_ms)
    df = dl.fetch_rest_klines(client, "BTCUSDT", "5m", base, now_ms + STEP, now_ms=now_ms)
    assert len(df) == 10


def test_funding_download(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    client = FakeClient({}, rest_until_ms=int(end.timestamp() * 1000))
    f = dl.download_funding(client, tmp_path, "BTCUSDT", start, end)
    assert len(f) == 365 * 3   # 3 начисления в сутки, 2024 — високосный, но end = 31.12 00:00
    assert f.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")   # +3 мс округлены
    assert f["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert sum("fundingRate" in c for c in client.calls) >= 2        # была пагинация


def test_quality_detects_gaps_and_bad_bars():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
    df = df.drop(idx[10:15])
    df.iloc[20, df.columns.get_loc("high")] = 98.0       # high < open
    df.iloc[30, df.columns.get_loc("volume")] = 0.0
    q = check_klines(df, "5m")
    assert q.missing_bars == 5
    assert q.largest_gaps[0][2] == 5
    assert q.ohlc_violations == 1 and q.zero_volume_bars == 1


def test_compare_daily_matches_resample():
    idx = pd.date_range("2024-01-01", periods=288 * 3, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    m5 = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1.0}, index=idx)
    d1 = m5.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out = compare_daily(m5, d1)
    assert out["days_compared"] == 3 and out["close_max_rel_diff"] == 0.0


def test_run_writes_quality_report(tmp_path, monkeypatch):
    now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    client = FakeClient({}, rest_until_ms=int(now.timestamp() * 1000))

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(dl, "datetime", _FixedDatetime)
    report = dl.run(["BTCUSDT"], ["5m"], datetime(2024, 1, 1, tzinfo=timezone.utc), tmp_path,
                    source="rest", client=client)
    sym = report["symbols"]["BTCUSDT"]
    assert sym["5m"]["bars"] == 2 * 288 and sym["5m"]["missing_bars"] == 0
    assert sym["funding"]["records"] == 6
    assert (tmp_path / "binance_um" / "quality_report.json").exists()
