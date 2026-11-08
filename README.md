# Walmart M5 Forecasting Baseline

This project compares 14-day and 28-day sales forecasts as inputs to a replenishment simulation. The working baseline uses one direct LightGBM Poisson model per horizon and fold, with 64 features.

| Model | Days 1–14 WAPE | Full-horizon WAPE |
|---|---:|---:|
| 14-day global baseline | 72.98% | 72.98% |
| 28-day global baseline | 73.13% | 74.38% |

The store × department benchmark achieved 72.16% and 73.57% WAPE over its full 14-day and 28-day horizons. The repository includes per-fold metrics for that benchmark, the 62-feature global comparison and the 64-feature baseline.

## Reproduce the backtests

Use Python 3.12 and install the pinned packages with `python -m pip install -r requirements.txt`. Place the three M5 input CSVs in `data/` locally. The original experiment used the [Nixtla mirror](https://nixtla.github.io/datasetsforecast/m5.html) of the [Kaggle M5 Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

```sh
python m5_lightgbm.py --sales data/sales_train_evaluation.csv --calendar data/calendar.csv --prices data/sell_prices.csv --all-stores --enriched --group-features --horizon 14 --threads 8 --output runs/baseline-14
python m5_lightgbm.py --sales data/sales_train_evaluation.csv --calendar data/calendar.csv --prices data/sell_prices.csv --all-stores --enriched --group-features --horizon 28 --threads 8 --output runs/baseline-28
```

Omit `--group-features` to reproduce the 62-feature global model. Replace it with `--grouped` to train the store, store × category and store × department benchmarks and their equal-weight average.

The forecast origins are `d_1857`, `d_1885` and `d_1913`, with 26 weekly training origins per fold. WAPE is total absolute error divided by total actual unit sales, averaged equally across folds. The planned-price scenario assumes target-week prices are known at forecast time. Sales are observed transactions. Inventory, purchase orders and lead times require separate simulation assumptions.

### Input SHA-256 checksums

| File | SHA-256 |
|---|---|
| sales_train_evaluation.csv | c21a519596680feb86f27a9e62f6c8b583f8be60c2c195f080ae8ca2990af2b7 |
| calendar.csv | 568d0fe5f41790142379698732908e4e57432c1c6396f3f59fb880a9c2b54231 |
| sell_prices.csv | 5c7cf045b1abbdc8d5007809c40b3682c5216075937f93e2cef01caddc4a30d9 |

The sales mirror omits the composite `id` column. All 30,490 product–store series and their 1,941 daily sales values were checked against a public copy whose MD5 matches the [Zenodo record](https://zenodo.org/records/10203108).

[PowerPoint summary](docs/Walmart-M5-Forecasting-Baseline.pptx)
