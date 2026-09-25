"""CryptoVladimirEngine — Vladimir Core (D → H1 → M5), адаптированный под крипту.

ВАЖНО: версия v0 написана по словесному описанию правил (docs/01_DECISIONS.md).
Когда будет доступен исходный код Vladimir Core, логика триггеров сверяется с
ним, а порт проверяется на данных FORTS (цель: 50–52 % винрейт, +35–45R/год).

Логика
------
D  — уровни: подтвержденные дневные пивоты 2/2 (strategies/levels.py),
     ATR(D) известный на начало дня. «День» = сутки UTC.
H1 — подтверждение (только завершенные H1-бары):
     • нет «ножа/палки»: ни один из последних h1_knife_lookback баров не длиннее
       h1_knife_max_atr × ATR;
     • и есть база (сжатие h1_base_bars баров в диапазоне ≤ h1_base_max_atr × ATR
       возле уровня) ИЛИ реакция (последний H1 коснулся уровня и закрылся за ним
       в сторону сделки).
M5 — триггер (на закрытии бара):
     • пробой: breakout_closes закрытий подряд за уровнем после закрытия с другой стороны;
     • реакция: касание → отскок ≥ bounce_min_atr → повторное касание с закрытием
       в сторону сделки (уровень не сломан закрытием дальше level_break_atr).
Вход — лимитный ордер у уровня (уровень ± entry_tolerance_atr), живет
order_ttl_bars баров, отменяется, если цена ушла к T1 без нас.
Выход — стоп (стоп-маркет, taker), цели T1..T3 частями (лимитки, maker),
выход по времени max_hold_bars (taker). Funding начисляется на каждой отметке
начисления, пока позиция открыта.

Консервативные допущения бэктеста
---------------------------------
• Лимитка исполняется, только если цена прошла СКВОЗЬ цену ордера (строго).
• Если на одном баре задеты и стоп, и цель — считаем, что первым был стоп.
• На баре входа цели не проверяются, стоп — проверяется.
• Стоп при гэпе исполняется по худшей цене (open бара), а не по цене стопа.
• Сигнал формируется на закрытии бара i, ордер активен с бара i+1.

Результат — в R (риск = |цена входа − стоп| на единицу позиции), с
раздельным учетом валового результата, комиссий, проскальзывания и funding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from crypto_bingx.backtest.costs import CostModel
from crypto_bingx.data.resample import resample_ohlcv
from crypto_bingx.strategies.levels import active_levels, atr_known_at_open, find_pivots
from crypto_bingx.strategies.params import VladimirParams

M5_NS = 5 * 60 * 1_000_000_000
H1_NS = 60 * 60 * 1_000_000_000


def _ns_index(df: pd.DataFrame) -> pd.DataFrame:
    """Индекс в наносекундах: pandas 3 может хранить время в us/ms, а арифметика
    ниже (asi8, M5_NS, H1_NS) рассчитана на наносекунды."""
    if df.index.unit != "ns":
        df = df.copy()
        df.index = df.index.as_unit("ns")
    return df


@dataclass
class Trade:
    symbol: str
    setup: str               # 'breakout' | 'reaction'
    side: int                # +1 long, -1 short
    level: float
    atr: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float       # номинальная цена лимитки
    stop_price: float
    targets: tuple
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""    # 'stop' | 'breakeven' | 'target' | 'time' | 'end'
    targets_hit: int = 0
    hold_bars: int = 0
    gross_r: float = 0.0     # по номинальным ценам, без издержек
    fee_r: float = 0.0
    slippage_r: float = 0.0
    funding_r: float = 0.0   # >0 — получили, <0 — заплатили
    net_r: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["targets"] = list(self.targets)
        return d


@dataclass
class _LevelState:
    """Состояние M5-автоматов для одного уровня."""
    last_side: int = 0        # сторона последнего закрытия: +1 выше, -1 ниже, 0 неизвестно
    streak: int = 0           # закрытий подряд на текущей стороне
    streak_from_cross: bool = False  # серия началась с пересечения уровня
    # реакция от поддержки (лонг) и от сопротивления (шорт): 0 ждем касания, 1 было касание, 2 был отскок
    long_state: int = 0
    long_since: int = -1
    short_state: int = 0
    short_since: int = -1


@dataclass
class _Pending:
    side: int
    setup: str
    level: float
    atr: float
    price: float
    stop: float
    targets: tuple
    placed_bar: int
    signal_time: pd.Timestamp


@dataclass
class _Position:
    trade: Trade
    entry_bar: int
    remaining: float = 1.0
    stop: float = 0.0
    risk: float = 0.0
    # денежные потоки на единицу позиции (в цене актива)
    gross: float = 0.0
    fees: float = 0.0
    slip: float = 0.0
    funding: float = 0.0
    fill_price: float = 0.0
    next_target: int = 0


@dataclass
class BacktestResult:
    symbol: str
    params: VladimirParams
    costs: CostModel
    trades: list = field(default_factory=list)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    signals: int = 0
    cancelled_orders: int = 0

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=list(Trade.__dataclass_fields__))
        return pd.DataFrame([t.to_dict() for t in self.trades])


class CryptoVladimirEngine:
    """Событийный бэктест одной монеты на M5.

    Пример:
        engine = CryptoVladimirEngine(VladimirParams(stop_atr_multiplier=0.45), CostModel.stress())
        result = engine.run(m5, symbol="BTCUSDT", funding=funding_df)
    """

    def __init__(self, params: VladimirParams | None = None, costs: CostModel | None = None):
        self.p = params or VladimirParams()
        self.c = costs or CostModel.base()

    # ------------------------------------------------------------------ подготовка

    def prepare(self, m5: pd.DataFrame, daily: pd.DataFrame | None = None) -> dict:
        """Дневной контекст (ATR, уровни) и H1-бары. Всё без заглядывания вперед."""
        p = self.p
        daily = daily if daily is not None else resample_ohlcv(m5, "1D")
        daily = daily.sort_index()
        atr = atr_known_at_open(daily, p.atr_period)
        pivots = find_pivots(daily, p.pivot_left, p.pivot_right)
        day_ctx: dict[int, tuple[float, np.ndarray]] = {}
        for day, a in atr.items():
            if not np.isfinite(a) or a <= 0:
                continue
            lv = active_levels(pivots, day, p.level_lookback_days, p.level_merge_atr * a)
            day_ctx[day.value] = (float(a), lv)
        h1 = resample_ohlcv(m5, "1h")
        return {"daily": daily, "atr": atr, "pivots": pivots, "day_ctx": day_ctx, "h1": h1}

    # ------------------------------------------------------------------ H1

    def _h1_confirms(self, side: int, level: float, atr: float, k: int, h1o, h1h, h1l, h1c) -> bool:
        """k — индекс последнего ЗАВЕРШЕННОГО H1-бара."""
        p = self.p
        if not p.h1_require_confirmation:
            return True
        need = max(p.h1_knife_lookback, p.h1_base_bars)
        if k < need - 1:
            return False
        # 1) Нет ножа/палки: импульсных баров без базы
        lo = k - p.h1_knife_lookback + 1
        if np.max(h1h[lo:k + 1] - h1l[lo:k + 1]) > p.h1_knife_max_atr * atr:
            return False
        # 2а) База: сжатие у уровня
        b0 = k - p.h1_base_bars + 1
        base_hi, base_lo = np.max(h1h[b0:k + 1]), np.min(h1l[b0:k + 1])
        prox = p.h1_base_proximity_atr * atr
        near_level = (base_lo - prox) <= level <= (base_hi + prox)
        if (base_hi - base_lo) <= p.h1_base_max_atr * atr and near_level:
            return True
        # 2б) Реакция: H1 провзаимодействовал с уровнем и закрылся за ним в сторону сделки
        if side > 0:
            return h1c[k] > level and h1l[k] <= level + prox
        return h1c[k] < level and h1h[k] >= level - prox

    # ------------------------------------------------------------------ основной цикл

    def run(self, m5: pd.DataFrame, symbol: str = "", funding: pd.DataFrame | None = None,
            daily: pd.DataFrame | None = None) -> BacktestResult:
        p, c = self.p, self.c
        m5 = _ns_index(m5.sort_index())
        daily = _ns_index(daily) if daily is not None else None
        ctx = self.prepare(m5, daily)
        res = BacktestResult(symbol=symbol, params=p, costs=c,
                             start=m5.index[0] if len(m5) else None,
                             end=m5.index[-1] if len(m5) else None)
        if len(m5) == 0:
            return res

        ts = m5.index.asi8
        o = m5["open"].to_numpy(float)
        h = m5["high"].to_numpy(float)
        l = m5["low"].to_numpy(float)
        cl = m5["close"].to_numpy(float)
        times = m5.index

        h1 = ctx["h1"]
        h1_ts = h1.index.as_unit("ns").asi8
        h1o, h1h, h1l, h1c = (h1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        # Индекс последнего завершенного H1 на момент закрытия каждого M5-бара:
        # H1-бар [t, t+1h) завершен, если t + 1h <= время закрытия M5 (open + 5m).
        last_h1 = np.searchsorted(h1_ts + H1_NS, ts + M5_NS, side="right") - 1

        day_id = (ts // (86_400 * 1_000_000_000)) * (86_400 * 1_000_000_000)
        hour = (ts // H1_NS) % 24

        funding_at: dict[int, float] = {}
        if funding is not None and c.use_funding and len(funding):
            f_ts = funding.index.as_unit("ns").floor("5min").asi8
            for t_, r_ in zip(f_ts, funding["funding_rate"].to_numpy(float)):
                if np.isfinite(r_):
                    funding_at[int(t_)] = float(r_)

        states: dict[float, _LevelState] = {}
        cur_day = None
        atr = np.nan
        levels = np.empty(0)
        trades_today = 0
        pending: _Pending | None = None
        pos: _Position | None = None

        for i in range(len(ts)):
            # ---------- смена суток: новый ATR и набор уровней
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                trades_today = 0
                atr, levels = ctx["day_ctx"].get(int(cur_day), (np.nan, np.empty(0)))
                states = {lv: states.get(lv, _LevelState()) for lv in levels}

            # ---------- funding по открытой позиции (начисляется на открытии бара-отметки)
            if pos is not None and i > pos.entry_bar and int(ts[i]) in funding_at:
                rate = funding_at[int(ts[i])]
                # лонг платит при положительной ставке, шорт получает
                pos.funding -= pos.trade.side * rate * o[i] * pos.remaining

            # ---------- лимитный ордер: исполнение / отмена
            if pending is not None and i > pending.placed_bar:
                filled = (l[i] < pending.price) if pending.side > 0 else (h[i] > pending.price)
                if filled:
                    pos = self._open_position(pending, i, times, symbol)
                    pending = None
                    trades_today += 1
                else:
                    t1 = pending.targets[0]
                    ran_away = (h[i] >= t1) if pending.side > 0 else (l[i] <= t1)
                    expired = i - pending.placed_bar >= p.order_ttl_bars
                    if ran_away or expired:
                        pending = None
                        res.cancelled_orders += 1

            # ---------- сопровождение позиции
            if pos is not None:
                closed = self._manage_position(pos, i, o, h, l, cl, times, is_last=(i == len(ts) - 1))
                if closed:
                    res.trades.append(pos.trade)
                    pos = None

            # ---------- M5-автоматы по уровням и новые сигналы
            if not np.isfinite(atr) or len(levels) == 0:
                continue
            signal = self._update_levels(states, i, h[i], l[i], cl[i], atr)
            if signal is None or pos is not None or pending is not None:
                continue
            if trades_today >= p.max_trades_per_day:
                continue
            if p.trade_hours_utc is not None:
                a, b = p.trade_hours_utc
                hr = hour[i]
                if not (a <= hr < b if a <= b else (hr >= a or hr < b)):
                    continue
            side, setup, level = signal
            if not self._h1_confirms(side, level, atr, int(last_h1[i]), h1o, h1h, h1l, h1c):
                continue
            res.signals += 1
            pending = self._make_order(side, setup, level, atr, cl[i], i, times[i])

        if pos is not None:   # на случай, если позиция открылась на последнем баре
            self._close_rest(pos, len(ts) - 1, cl[-1], "end", times, maker=False)
            res.trades.append(pos.trade)
        return res

    # ------------------------------------------------------------------ M5-автоматы

    def _update_levels(self, states: dict, i: int, hi: float, lo: float, close: float,
                       atr: float):
        """Обновить состояния всех уровней по закрытию бара i; вернуть сигнал
        (side, setup, level) ближайшего к цене уровня или None."""
        p = self.p
        touch = p.touch_tolerance_atr * atr
        bounce = p.bounce_min_atr * atr
        brk = p.level_break_atr * atr
        best = None
        best_dist = np.inf
        for lv, st in states.items():
            side = 1 if close > lv else (-1 if close < lv else 0)
            # --- пробой: серия закрытий по одну сторону уровня
            if side != 0:
                if side == st.last_side:
                    st.streak += 1
                else:
                    st.streak_from_cross = st.last_side == -side
                    st.streak = 1
                st.last_side = side
            sig = None
            if (p.enable_breakout and side != 0 and st.streak_from_cross
                    and st.streak == p.breakout_closes):
                sig = (side, "breakout", lv)

            # --- реакция от поддержки (цена над уровнем) → лонг
            if p.enable_reaction:
                if close < lv - brk or (st.long_state and i - st.long_since > p.reaction_max_bars):
                    st.long_state = 0
                touched_from_above = lo <= lv + touch and close > lv
                if st.long_state == 0 and touched_from_above:
                    st.long_state, st.long_since = 1, i
                elif st.long_state == 1 and close >= lv + bounce:
                    st.long_state = 2
                elif st.long_state == 2 and touched_from_above:
                    st.long_state = 0
                    sig = sig or (1, "reaction", lv)
                # --- реакция от сопротивления (цена под уровнем) → шорт
                if close > lv + brk or (st.short_state and i - st.short_since > p.reaction_max_bars):
                    st.short_state = 0
                touched_from_below = hi >= lv - touch and close < lv
                if st.short_state == 0 and touched_from_below:
                    st.short_state, st.short_since = 1, i
                elif st.short_state == 1 and close <= lv - bounce:
                    st.short_state = 2
                elif st.short_state == 2 and touched_from_below:
                    st.short_state = 0
                    sig = sig or (-1, "reaction", lv)

            if sig is not None:
                d = abs(close - lv)
                if d < best_dist:
                    best, best_dist = sig, d
        return best

    # ------------------------------------------------------------------ ордера и позиции

    def _make_order(self, side: int, setup: str, level: float, atr: float, close: float,
                    i: int, t: pd.Timestamp) -> _Pending:
        p = self.p
        tol = p.entry_tolerance_atr * atr
        # Лимитка у уровня, но не «по рынку»: для лонга не выше текущей цены, для шорта не ниже.
        price = min(level + tol, close) if side > 0 else max(level - tol, close)
        stop = price - side * p.stop_atr_multiplier * atr
        targets = tuple(price + side * m * atr for m in p.target_atr_multipliers)
        return _Pending(side=side, setup=setup, level=level, atr=atr, price=price, stop=stop,
                        targets=targets, placed_bar=i, signal_time=t)

    def _open_position(self, od: _Pending, i: int, times, symbol: str) -> _Position:
        c = self.c
        fill = od.price * (1 + od.side * c.slippage_limit)
        trade = Trade(symbol=symbol, setup=od.setup, side=od.side, level=od.level, atr=od.atr,
                      signal_time=od.signal_time, entry_time=times[i], entry_price=od.price,
                      stop_price=od.stop, targets=od.targets)
        pos = _Position(trade=trade, entry_bar=i, stop=od.stop, risk=abs(od.price - od.stop),
                        fill_price=fill)
        pos.fees += c.maker_fee * fill
        pos.slip += abs(fill - od.price)
        return pos

    def _exit_part(self, pos: _Position, qty: float, nominal: float, maker: bool, bar_open: float | None = None):
        """Закрыть часть позиции. nominal — цена ордера; для стопа при гэпе берется худшая цена."""
        c = self.c
        side = pos.trade.side
        px = nominal
        if bar_open is not None:
            # стоп-маркет при гэпе исполняется по open, если он хуже цены стопа
            px = min(nominal, bar_open) if side > 0 else max(nominal, bar_open)
        fill = px * (1 - side * c.slippage(maker))
        pos.gross += side * (nominal - pos.trade.entry_price) * qty
        pos.slip += (abs(fill - nominal)) * qty
        pos.fees += c.fee(maker) * fill * qty
        pos.remaining -= qty

    def _close_rest(self, pos: _Position, i: int, price: float, reason: str, times,
                    maker: bool, bar_open: float | None = None):
        self._exit_part(pos, pos.remaining, price, maker, bar_open)
        pos.remaining = 0.0
        self._finalize(pos, i, reason, times)

    def _finalize(self, pos: _Position, i: int, reason: str, times):
        t = pos.trade
        t.exit_time = times[i]
        t.exit_reason = reason
        t.hold_bars = i - pos.entry_bar
        r = pos.risk if pos.risk > 0 else np.nan
        t.gross_r = pos.gross / r
        t.fee_r = pos.fees / r
        t.slippage_r = pos.slip / r
        t.funding_r = pos.funding / r
        t.net_r = t.gross_r - t.fee_r - t.slippage_r + t.funding_r

    def _manage_position(self, pos: _Position, i: int, o, h, l, cl, times, is_last: bool) -> bool:
        """Обработать бар i для открытой позиции. True — позиция закрыта."""
        p = self.p
        t = pos.trade
        side = t.side
        # 1) Стоп (проверяется первым — консервативно, в т.ч. на баре входа)
        stop_hit = (l[i] <= pos.stop) if side > 0 else (h[i] >= pos.stop)
        if stop_hit:
            reason = "breakeven" if t.targets_hit > 0 and p.breakeven_after_t1 else "stop"
            gap_open = o[i] if i > pos.entry_bar else None
            self._close_rest(pos, i, pos.stop, reason, times, maker=False, bar_open=gap_open)
            return True
        # 2) Цели — со следующего бара после входа
        if i > pos.entry_bar:
            while pos.next_target < len(t.targets):
                tgt = t.targets[pos.next_target]
                if not ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                    break
                qty = p.target_fractions[pos.next_target]
                if pos.next_target == len(t.targets) - 1:
                    qty = pos.remaining
                self._exit_part(pos, qty, tgt, maker=True)
                pos.next_target += 1
                t.targets_hit = pos.next_target
                if pos.next_target == 1 and p.breakeven_after_t1:
                    pos.stop = t.entry_price
            if pos.remaining <= 1e-12:
                self._finalize(pos, i, "target", times)
                return True
        # 3) Выход по времени / конец данных
        if i - pos.entry_bar >= p.max_hold_bars:
            self._close_rest(pos, i, cl[i], "time", times, maker=False)
            return True
        if is_last:
            self._close_rest(pos, i, cl[i], "end", times, maker=False)
            return True
        return False
