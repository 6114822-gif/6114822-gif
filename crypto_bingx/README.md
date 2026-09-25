# crypto_bingx — Vladimir Core для BingX USDT-M

Исследовательский проект: перенос методологии Vladimir Core (D → H1 → M5) с FORTS
на бессрочные фьючерсы BingX. Сейчас идет **этап 1: исследование**.

- План: [docs/00_PLAN.md](docs/00_PLAN.md)
- Журнал решений: [docs/01_DECISIONS.md](docs/01_DECISIONS.md)
- Отчеты: [docs/reports/](docs/reports/)

## Структура

```
crypto_bingx/
├── config.py                      пути и значения по умолчанию
├── data/
│   ├── download_binance_data.py   загрузка свечей и funding (архив + REST), докачка
│   ├── storage.py                 Parquet-хранилище
│   ├── quality.py                 проверка качества: пропуски, дубли, битые OHLC, выбросы
│   ├── resample.py                M5 → H1 → D1 по UTC
│   └── store/                     данные (не в git)
├── strategies/
│   ├── params.py                  VladimirParams: все параметры стратегии
│   ├── levels.py                  пивоты D 2/2, ATR без заглядывания вперед
│   └── crypto_vladimir_engine.py  CryptoVladimirEngine: сигналы + событийный бэктест
├── backtest/
│   ├── costs.py                   CostModel: maker/taker/проскальзывание/funding
│   ├── metrics.py                 винрейт, PF, R, max DD, Sharpe, разбивка по годам
│   └── run_backtest.py            прогон по сетке параметров
├── research/
│   └── cost_vs_stop.py            шаг 1.2: издержки против размера стопа
├── tests/                         pytest, работают без сети
└── docs/
```

## Установка

Нужен Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r crypto_bingx/requirements.txt
```

Все команды запускаются **из корня репозитория** (из папки, где лежит `crypto_bingx/`).

## Порядок работы на этапе 1

### 1. Скачать данные

```bash
# BTC/ETH/SOL, M5 + D1, последние 3 года + funding
python -m crypto_bingx.data.download_binance_data

# рекомендую с 2021-01-01: захватывает бычий 2021 и медвежий 2022
python -m crypto_bingx.data.download_binance_data --start 2021-01-01
```

- Завершенные месяцы берутся из архива `data.binance.vision` с проверкой SHA-256.
- Текущий месяц и funding берутся через REST `fapi.binance.com`. Ключи не нужны.
- Повторный запуск докачивает только новые бары.
- Отчет о качестве сохраняется в `crypto_bingx/data/store/binance_um/quality_report.json`.
- Объем данных: около 30–40 МБ Parquet на монету за 3 года M5.

Если `data.binance.vision` недоступен, добавьте `--source rest` (медленнее, но работает).
Если недоступен REST `fapi.binance.com` (например, Binance отвечает 451 серверам из США), загрузчик сам переключится на архивы: хвост свечей возьмёт из дневных архивов (с задержкой ~1 сутки), funding — из месячного архива.
Если Binance отвечает 403, домен закрыт сетевой политикой или Binance блокирует регион сервера.

### 2. Проверить гипотезу про издержки

```bash
python -m crypto_bingx.research.cost_vs_stop
```

Отчет сохраняется в `docs/reports/cost_vs_stop.md`: ATR(D) в % по годам и издержки в R
для стопов 0.15 / 0.3 / 0.45 / 0.6 ATR.

### 3. Прогнать сетку стратегии

```bash
# H0 (FORTS как есть) + стоп 0.3/0.45/0.6 ATR × цели FORTS ×2/×3, издержки base и stress
python -m crypto_bingx.backtest.run_backtest --forts-baseline --jobs 4
```

Сводка сохраняется в `docs/reports/grid_binance_*.md`, сделки — в `data/store/results/`.

### Тесты

```bash
python -m pytest crypto_bingx/tests -q
```

Тесты не ходят в сеть. Они проверяют:
- загрузчик на поддельных ответах Binance;
- точные расчеты движка на заданных вручную сценариях;
- отсутствие заглядывания в будущее: обрезка данных не меняет прошлые сделки;
- отсутствие ложного преимущества на случайном блуждании.

## Как читать результаты

- **R** — результат сделки в единицах риска. Риск = |цена входа − стоп|. +2R означает прибыль в два риска.
- `gross_total_r` — результат без издержек, то есть чистое преимущество сигнала.
- `avg_cost_r` — средние издержки на сделку в R (комиссии + проскальзывание − funding).
  На FORTS это ≈ 0.01–0.02R. Здесь это главный показатель.
- `cost_share_of_gross` — какую долю валовой прибыли съели издержки.
- `max_dd_pct`, `cagr_pct`, `sharpe` считаются при риске 1 % на сделку (`--risk`).
  Это оценка по одному инструменту; портфельные лимиты (−2 % в день, −5 % в неделю, −15 % DD)
  подключаются в модуле риска на этапе 2.

## Ограничения версии v0

- Правила H1 и M5 формализованы по словесному описанию (см. журнал решений, п. 1–6).
  После получения исходного кода Vladimir Core логика сверяется с ним, а порт проверяется
  на данных FORTS.
- Внутри M5-бара порядок цен неизвестен, поэтому при неоднозначности бэктест считает худший исход.
  Уточнение на M1 возможно позже.
- Бэктест идет по одной монете, без портфельного риск-менеджмента.
