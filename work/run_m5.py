from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "Time-LLM"))
from models.TimeLLM import Model

def scores(meta, history, actual, predictions):
    weights = pd.read_csv(root / "m5/weights_evaluation.csv").set_index(
        ["Level_id", "Agg_Level_1", "Agg_Level_2"]
    )["weight"]
    levels = [(), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",),
              ("state_id", "cat_id"), ("state_id", "dept_id"),
              ("store_id", "cat_id"), ("store_id", "dept_id"), ("item_id",),
              ("state_id", "item_id"), ("item_id", "store_id")]
    result = []
    for i, keys in enumerate(levels, 1):
        a = meta[keys[0]] if keys else ["Total"] * len(meta)
        b = meta[keys[1]] if len(keys) == 2 else ["X"] * len(meta)
        codes, groups = pd.factorize(pd.MultiIndex.from_arrays([a, b]))
        w = np.array([weights.loc[(f"Level{i}", *group)] for group in groups])
        assert abs(w.sum() - 1) < 1e-5
        agg = csr_matrix((np.ones(len(meta)), (codes, np.arange(len(meta)))))
        h = agg @ history
        active = h.cumsum(axis=1)[:, :-1] > 0
        scale = (np.diff(h, axis=1)**2 * active).sum(axis=1) / active.sum(axis=1).clip(1)
        valid = w > 0
        assert (scale[valid] > 0).all()
        result.append([float(w[valid] @ np.sqrt(((agg @ (p - actual))**2).mean(axis=1)[valid] / scale[valid]))
                       for p in predictions])
    return np.array(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    calendar = pd.read_csv(root / 'm5/calendar.csv', parse_dates=['date']).date
    assert len(calendar) >= 1969 and calendar.iloc[1941] == pd.Timestamp('2016-05-23')
    dtype = {f"d_{d}": np.int16 for d in range(1, 1942)}
    data = pd.read_csv(root / "m5/sales_train_evaluation.csv", dtype=dtype)
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    meta = data[cols]
    history = data[list(dtype)].to_numpy()
    del data
    test = pd.read_csv(root / "m5/sales_test_evaluation.csv")
    actual = meta.merge(test, on=cols, validate="one_to_one", sort=False).iloc[:, 5:].to_numpy()
    assert history.shape == (30490, 1941) and actual.shape == (30490, 28)
    naive = np.tile(history[:, -7:], (1, 4))
    baseline = scores(meta, history, actual, [naive])
    print(f"Weekly naive WRMSSE: {baseline.mean():.6f}", flush=True)

    context = 56
    config = SimpleNamespace(task_name="short_term_forecast", pred_len=28,
        seq_len=context, d_ff=16, llm_dim=768, patch_len=16, stride=8,
        llm_model="GPT2", llm_layers=1, prompt_domain=True,
        content="Daily unit sales of a product at a Walmart store. ",
        dropout=0.05, d_model=16, n_heads=2, enc_in=1)
    model = Model(config)
    model.llm_model.eval()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    offsets = np.arange(context + 28)
    for step in range(args.steps):
        rows = rng.integers(len(history), size=args.batch)
        starts = rng.integers(history.shape[1] - len(offsets) + 1, size=args.batch)
        samples = torch.from_numpy(history[rows[:, None], starts[:, None] + offsets]).float().unsqueeze(-1)
        x, y = samples[:, :context], samples[:, context:]
        opt.zero_grad()
        p = model(x, None, None, None)
        loss = ((p - y) / x.std(dim=1, keepdim=True).clamp_min(1)).square().mean()
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 8 == 0:
            print(f"Train {step + 1}/{args.steps}: loss={loss.item():.5f}", flush=True)

    model.eval()
    prediction = []
    batch = args.batch
    with torch.inference_mode():
        source = model.mapping_layer(model.word_embeddings.T)
        model.mapping_layer.forward = lambda _: source
        for i in range(0, len(history), batch):
            x = torch.from_numpy(history[i:i + batch, -context:].copy()).float().unsqueeze(-1)
            prediction.append(model(x, None, None, None).clamp_min(0).squeeze(-1).numpy())
            if i == 0 or (i // batch + 1) % 64 == 0:
                print(f"Inference {min(i + batch, len(history))}/{len(history)}", flush=True)
    forecast = np.concatenate(prediction)
    np.save(root / 'm5-forecast.npy', forecast)
    levels = scores(meta, history, actual, [naive, forecast])
    out = root.parent / "outputs"
    report = pd.DataFrame(levels, columns=["weekly_naive", "time_llm"],
                          index=[f"Level{i}" for i in range(1, 13)])
    report.loc["WRMSSE"] = report.mean()
    dates = calendar.iloc[1941:1969].reset_index(drop=True)
    daily = pd.DataFrame({"date": dates.to_numpy(), "actual": actual.sum(axis=0),
                          "forecast": forecast.sum(axis=0, dtype=np.float64)})
    daily["residual"] = daily["forecast"] - daily["actual"]
    daily.to_csv(out / "m5_daily_residuals.csv", index=False)
    pd.DataFrame({"item_id": np.repeat(meta.item_id.to_numpy(), 28),
                  "store_id": np.repeat(meta.store_id.to_numpy(), 28),
                  "date": np.tile(dates.to_numpy(), len(meta)),
                  "actual": actual.ravel(), "forecast": forecast.ravel(),
                  "residual": (forecast - actual).ravel()}).to_csv(
        out / "m5_item_store_residuals.csv", index=False, float_format="%.6f")
    report.to_csv(out / "m5_wrmsse.csv", index_label="level")
    print(report.to_string(), flush=True)

if __name__ == "__main__":
    main()



