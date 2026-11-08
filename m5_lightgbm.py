"""Backtest direct multi-horizon LightGBM forecasts on M5 sales."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def load_context(table, calendar_path, prices_path, group_features=False):
    calendar = pd.read_csv(calendar_path, parse_dates=["date"]).sort_values("date")
    if "d" in calendar and calendar["d"].tolist() != [
        f"d_{i}" for i in range(1, len(calendar) + 1)
    ]:
        raise ValueError("Unexpected calendar day identifiers")
    if not calendar["date"].equals(pd.Series(pd.date_range("2011-01-29", periods=len(calendar)),
                                             index=calendar.index)):
        raise ValueError("Expected a continuous M5 calendar starting on 2011-01-29")
    categories = {
        name: pd.factorize(table[name], sort=True)[0]
        for name in ("item_id", "dept_id", "cat_id", "store_id", "state_id")
    }
    if group_features:
        for name in ("cat_id", "dept_id"):
            groups = pd.MultiIndex.from_frame(table[["store_id", name]])
            categories[f"store_{name}"] = groups.factorize(sort=True)[0]
    dates = {
        name: pd.factorize(calendar[name], sort=True)[0] + 1
        for name in ("event_name_1", "event_type_1", "event_name_2", "event_type_2")
    }
    for name in ("month", "year"):
        dates[name] = calendar[name].to_numpy()
    dates["day_of_month"] = calendar["date"].dt.day.to_numpy()
    week_index, weeks = pd.factorize(calendar["wm_yr_wk"], sort=True)
    prices = pd.read_csv(prices_path, dtype={"sell_price": np.float32})
    keys = ["item_id", "store_id"]
    indexed_series = table[keys].assign(series=np.arange(len(table)))
    prices = prices.merge(indexed_series, on=keys, validate="many_to_one")
    week_columns = weeks.get_indexer(prices["wm_yr_wk"])
    if (week_columns < 0).any() or prices.duplicated(keys + ["wm_yr_wk"]).any():
        raise ValueError("Unexpected week or duplicate in price table")
    price_grid = np.full((len(table), len(weeks)), np.nan, dtype=np.float32)
    price_grid[prices["series"].to_numpy(), week_columns] = prices["sell_price"].to_numpy()
    states = ["CA", "TX", "WI"]
    state_index = pd.Index(states).get_indexer(table["state_id"])
    if (state_index < 0).any():
        raise ValueError("Unexpected state in M5 sales")
    snap = calendar[[f"snap_{state}" for state in states]].to_numpy()
    return categories, dates, week_index, price_grid, state_index, snap


def features(sales, origin, horizon, context=None, enriched=False):
    series_count = len(sales)
    lead = np.tile(np.arange(1, horizon + 1), series_count)
    series = np.repeat(np.arange(series_count), horizon)
    columns = {
        "series": series,
        "lead": lead,
        "weekday": (origin + lead) % 7,
    }
    for lag in (0, 6, 13, 27):
        columns[f"lag_{lag}"] = sales[series, origin - lag]
    windows = (*range(1, 29), 56) if enriched else (7, 14, 28, 56)
    history = sales[:, origin - 27 : origin + 1]
    if enriched:
        sums = np.cumsum(history[:, ::-1], axis=1, dtype=np.float64)
    for window in windows:
        if enriched and window <= 28:
            mean = sums[:, window - 1] / window
        else:
            mean = sales[:, origin - window + 1 : origin + 1].mean(axis=1)
        columns[f"mean_{window}"] = mean[series]
    if enriched:
        weekday = columns["weekday"]
        # M5 d_1 is Saturday, so 0 and 1 are Saturday and Sunday.
        weekend = weekday < 2
        historical_weekday = np.arange(origin - 27, origin + 1) % 7
        historical_weekend = historical_weekday < 2
        same_day = np.column_stack([
            history[:, historical_weekday == day].mean(axis=1) for day in range(7)
        ])
        columns["same_weekday_mean_28"] = same_day[series, weekday]
        columns["same_weekday_latest"] = sales[series, origin - (origin - weekday) % 7]
        columns["target_is_weekend"] = weekend
        columns["weekday_x_origin_sales"] = (weekday + 1) * columns["lag_0"]
        columns["weekend_x_origin_sales"] = weekend * columns["lag_0"]
        columns["workday_x_origin_sales"] = (~weekend) * columns["lag_0"]
        columns["target_day_type_mean_28"] = np.where(
            weekend,
            history[:, historical_weekend].mean(axis=1)[series],
            history[:, ~historical_weekend].mean(axis=1)[series],
        )
        columns["trend_mean_7_vs_28"] = columns["mean_7"] - columns["mean_28"]
        columns["std_28"] = history.std(axis=1)[series]
        columns["zero_sales_share_28"] = (history == 0).mean(axis=1)[series]
    if context is not None:
        categories, dates, weeks, prices, states, snap = context
        target = origin + lead
        columns.update({name: values[series] for name, values in categories.items()})
        columns.update({name: values[target] for name, values in dates.items()})
        columns["snap"] = snap[target, states[series]]
        columns["planned_price"] = prices[series, weeks[target]]
        columns["origin_price"] = prices[series, weeks[origin]]
        columns["price_change"] = columns["planned_price"] / columns["origin_price"] - 1
    return pd.DataFrame(columns, dtype=np.float32)


def scores(actual, predicted):
    error = predicted - actual
    volume = float(actual.sum())
    return {
        "mae": float(np.abs(error).mean()),
        "wape": float(np.abs(error).sum()) / volume if volume else None,
        "cumulative_mae_per_series": float(np.abs(error.sum(axis=1)).mean()),
    }


def fit_model(sales, train_origins, horizon, context, enriched, categorical, threads):
    first = features(sales, train_origins[0], horizon, context, enriched)
    block_size = len(first)
    matrix = np.empty((block_size * len(train_origins), len(first.columns)), dtype=np.float32)
    matrix[:block_size] = first.to_numpy()
    columns = first.columns
    del first
    for block, i in enumerate(train_origins[1:], 1):
        matrix[block * block_size : (block + 1) * block_size] = features(
            sales, i, horizon, context, enriched
        ).to_numpy()
    x = pd.DataFrame(matrix, columns=columns, copy=False)
    del matrix
    y = np.concatenate(
        [sales[:, i + 1 : i + 1 + horizon].ravel() for i in train_origins]
    )
    model = lgb.train(
        {
            "objective": "poisson",
            "learning_rate": 0.05,
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": threads,
            "seed": 42,
            "verbosity": -1,
        },
        lgb.Dataset(x, label=y, categorical_feature=categorical),
        num_boost_round=150,
    )
    del x, y
    return model


def subset_context(context, indices):
    categories, dates, weeks, prices, states, snap = context
    return (
        {name: values[indices] for name, values in categories.items()},
        dates, weeks, prices[indices], states[indices], snap,
    )


def run(args):
    table = pd.read_csv(args.sales)
    store = None if args.all_stores else args.store
    if store:
        table = table.loc[table["store_id"] == store]
    series_ids = table["id"] if "id" in table else table["item_id"] + "_" + table["store_id"]
    days = sorted(
        (c for c in table if c.startswith("d_")), key=lambda c: int(c[2:])
    )
    if days != [f"d_{day}" for day in range(1, len(days) + 1)]:
        raise ValueError("Expected contiguous M5 columns beginning at d_1")
    sales = table[days].to_numpy(dtype=np.float32)
    if not len(sales) or args.folds < 1 or args.train_weeks < 1:
        raise ValueError("Select a nonempty store and positive fold/training counts")
    args.output.mkdir(parents=True, exist_ok=True)
    context = (
        load_context(table, args.calendar, args.prices, args.group_features)
        if args.calendar else None
    )
    categorical = ["series", "weekday"]
    if context is not None:
        categorical.extend(context[0])
        categorical.extend(name for name in context[1] if name.startswith("event_"))
    partitions = {}
    if args.grouped:
        for columns in (["store_id"], ["store_id", "cat_id"], ["store_id", "dept_id"]):
            partitions["+".join(columns)] = list(table.groupby(columns, sort=True).indices.values())
    feature_count = len(features(sales, 55, 1, context, args.enriched).columns) if len(days) > 56 else 0
    del table
    reports = []
    importances = []
    fold_count = 1 if args.forecast_final else args.folds
    for fold in range(fold_count):
        # Holdout origins and the last eligible training origin match across horizons.
        origin = len(days) - 1 if args.forecast_final else len(days) - 29 - 28 * (args.folds - 1 - fold)
        if args.forecast_final and (days[origin] != "d_1941" or args.horizon != 28):
            raise ValueError("Final forecast requires the M5 d_1941 cutoff and a 28-day horizon")
        last_train_origin = origin - 28
        train_origins = range(
            max(55, last_train_origin - 7 * (args.train_weeks - 1)),
            last_train_origin + 1,
            7,
        )
        if not train_origins:
            raise ValueError("Not enough historical sales for the requested backtest")
        print(f"Training fold {fold + 1}/{fold_count} at {days[origin]}", flush=True)
        horizon = args.horizon
        actual = sales[:, origin + 1 : origin + 1 + horizon]
        prediction_shape = (len(sales), horizon)
        grouped_predictions = {}
        if args.grouped:
            for level, groups in partitions.items():
                print(f"  {level}: {len(groups)} models", flush=True)
                level_prediction = np.empty(prediction_shape, dtype=np.float64)
                for indices in groups:
                    group_sales = sales[indices]
                    group_context = subset_context(context, indices)
                    model = fit_model(
                        group_sales, train_origins, horizon, group_context,
                        args.enriched, categorical, args.threads,
                    )
                    group_features = features(
                        group_sales, origin, horizon, group_context, args.enriched
                    )
                    level_prediction[indices] = model.predict(group_features).reshape(-1, horizon)
                grouped_predictions[level] = level_prediction
            predicted = np.mean(list(grouped_predictions.values()), axis=0)
        else:
            model = fit_model(
                sales, train_origins, horizon, context, args.enriched, categorical, args.threads
            )
            predicted = model.predict(
                features(sales, origin, horizon, context, args.enriched)
            ).reshape(prediction_shape)
            gain = model.feature_importance(importance_type="gain")
            gain_total = gain.sum()
            importances.append(pd.DataFrame({
                "origin": days[origin],
                "feature": model.feature_name(),
                "gain": gain,
                "gain_share": gain / gain_total if gain_total else np.zeros_like(gain),
                "split_count": model.feature_importance(importance_type="split"),
            }))
            pd.concat(importances, ignore_index=True).to_csv(
                args.output / "feature_importance_by_fold.csv", index=False
            )
        if args.forecast_final:
            final = pd.DataFrame(predicted, columns=[f"F{i}" for i in range(1, horizon + 1)])
            final.insert(0, "id", series_ids.str.removesuffix("_evaluation").to_numpy() + "_evaluation")
            final.to_csv(args.output / "final_forecast.csv", index=False)
            return
        seasonal = sales[:, origin - 6 + np.arange(horizon) % 7]
        report = {
            "origin": days[origin],
            "common_14d": scores(actual[:, :14], predicted[:, :14]),
            "common_14d_seasonal": scores(actual[:, :14], seasonal[:, :14]),
            "full_horizon": scores(actual, predicted),
            "full_horizon_seasonal": scores(actual, seasonal),
        }
        if horizon == 28:
            report["days_15_to_28"] = scores(actual[:, 14:], predicted[:, 14:])
            report["days_15_to_28_seasonal"] = scores(actual[:, 14:], seasonal[:, 14:])
        if grouped_predictions:
            report["by_training_group"] = {
                name: {
                    "common_14d": scores(actual[:, :14], forecast[:, :14]),
                    "full_horizon": scores(actual, forecast),
                    **({"days_15_to_28": scores(actual[:, 14:], forecast[:, 14:])}
                       if horizon == 28 else {}),
                }
                for name, forecast in grouped_predictions.items()
            }
        reports.append(report)
        pd.DataFrame({
            "id": np.repeat(series_ids.to_numpy(), horizon),
            "day": np.tile(days[origin + 1 : origin + 1 + horizon], len(sales)),
            "actual_sales": actual.ravel(),
            "prediction": predicted.ravel(),
            "seasonal_naive": seasonal.ravel(),
            **{f"prediction_{name}": forecast.ravel() for name, forecast in grouped_predictions.items()},
        }).to_csv(args.output / f"predictions_{days[origin]}.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps({
            "horizon": args.horizon,
            "store": store,
            "series_count": len(sales),
            "sales_file": args.sales.name,
            "calendar_file": args.calendar.name if args.calendar else None,
            "prices_file": args.prices.name if args.prices else None,
            "price_mode": "planned_price_known" if args.prices else None,
            "enriched_features": args.enriched,
            "group_features": args.group_features,
            "feature_count": feature_count,
            "models_per_fold": {name: len(groups) for name, groups in partitions.items()},
            "training_weeks": args.train_weeks,
            "lightgbm_version": lgb.__version__,
            "folds": reports,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", required=True, type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--enriched", action="store_true")
    training = parser.add_mutually_exclusive_group()
    training.add_argument("--grouped", action="store_true")
    training.add_argument("--group-features", action="store_true")
    parser.add_argument("--forecast-final", action="store_true")
    parser.add_argument("--horizon", required=True, type=int, choices=(14, 28))
    parser.add_argument("--output", required=True, type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--store", default="CA_1")
    scope.add_argument("--all-stores", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-weeks", type=int, default=26)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if bool(args.calendar) != bool(args.prices):
        parser.error("--calendar and --prices must be supplied together")
    if args.grouped and not (args.all_stores and args.calendar):
        parser.error("--grouped requires --all-stores, --calendar and --prices")
    if args.group_features and not args.calendar:
        parser.error("--group-features requires --calendar and --prices")
    run(args)
