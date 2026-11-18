import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import event_metrics
from retail_patch import PatchTransformer, make_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["favorita"])
    parser.add_argument("mode", choices=["long"])
    args = parser.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    torch_device = "cuda"
    weekly = args.dataset == "dominicks"
    block = 8 if weekly else 28
    history = 4 if weekly else 28
    long = args.mode in ("long", "daily3y")
    daily = args.mode == "daily3y"
    tokens_per_patch = 1 if (weekly or daily) else 7
    if long and weekly:
        parser.error("Long context is configured for daily data.")
    patch_count = 1099 if daily else (157 if long else 4)
    context = 1099 if long else history
    batch_size = 160 if daily else 1024 if args.mode in ("long", "short12", "daily3y") else 4096
    parquet = pq.ParquetFile(f"work/{args.dataset}-panel.parquet")
    date_based = args.dataset == "favorita"

    def periods(col):
        return col.to_numpy(zero_copy_only=False).astype("datetime64[D]").astype(np.int64) if date_based else col.to_numpy().astype(np.int64)

    # Scan only the period column first to size the panel.
    low = min(periods(batch.column(0)).min() for batch in parquet.iter_batches(columns=["period"]))
    high = max(periods(batch.column(0)).max() for batch in parquet.iter_batches(columns=["period"]))
    span = int(high-low+1)
    # Detect new series in sorted input, carrying the last pair across chunks.
    ids = []
    prior = None
    for batch in parquet.iter_batches(columns=["store", "product"]):
        pair = np.column_stack([batch.column(i).to_numpy() for i in range(2)])
        breaks = np.r_[prior is None or np.any(pair[0] != prior), np.any(pair[1:] != pair[:-1], axis=1)]
        ids.append(pair[breaks])
        prior = pair[-1]
    ids = np.concatenate(ids)
    assert len(ids) == len(np.unique(ids, axis=0))
    # Zero daily transactions are observed sample zeros after the first recorded
    # sale. Unreported scanner weeks stay missing.
    raw = np.full((len(ids), span), np.nan if weekly else 0, np.float32)
    first = np.full(len(ids), span, dtype=np.int32)
    cursor = -1
    prior = None
    for batch in parquet.iter_batches(columns=["store", "product", "period", "sales"]):
        pair = np.column_stack([batch.column(i).to_numpy() for i in range(2)])
        breaks = np.r_[prior is None or np.any(pair[0] != prior), np.any(pair[1:] != pair[:-1], axis=1)]
        row = cursor + np.cumsum(breaks)
        t = periods(batch.column(2))-low
        raw[row,t] = np.maximum(batch.column(3).to_numpy(),0)
        np.minimum.at(first,row,t)
        prior, cursor = pair[-1], row[-1]
    assert cursor == len(ids)-1
    fit_end = span-2*block
    test = span-block
    seen = first < fit_end
    scale = float(np.nansum(raw[:, :fit_end], dtype=np.float64) /
                  np.maximum(fit_end-first,0).sum())
    # Keep the potentially large panel in host memory. Only batch targets use CUDA.
    cal = make_dates(span, torch_device, weekly=weekly)
    offsets = torch.arange(-context,0,device=torch_device)
    model = PatchTransformer(cal.shape[1],len(ids),args.mode!="date", tokens_per_patch, sales_history=args.mode in ("history", "long", "short12", "daily3y"), patch_count=patch_count, layers=12 if args.mode in ("long", "short12", "daily3y") else None).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=0.001,fused=True)
    tag = f"{args.dataset}-patch-{args.mode}"
    rows_per_block = max(1, 1_000_000 // fit_end)
    groups = []
    for base in range(0,len(ids),rows_per_block):
        end = min(len(ids),base+rows_per_block)
        eligible = np.arange(fit_end) >= np.maximum(history,first[base:end,None]+1)
        if weekly:
            eligible &= np.isfinite(raw[base:end,:fit_end])
        series, times = np.where(eligible)
        if len(series):
            groups.append((series.astype(np.int32)+base,times.astype(np.int32)))
    print(f"{tag}: {len(ids):,} series, {sum(len(x) for x,_ in groups):,} fit windows, "
          f"{sum(p.numel() for p in model.parameters()):,} trainable parameters", flush=True)
    random = np.random.default_rng(42)

    lag_offsets = np.arange(-context, 0)

    def pred(series, t):
        past = t[:, None] + lag_offsets
        observed_start = first[series, None]
        padding = None
        inputs = None
        if args.mode in ("history", "long", "short12", "daily3y"):
            values = raw[series[:, None], np.maximum(past, 0)]
            values[past < observed_start] = np.nan
            inputs = torch.as_tensor(values / scale, device=torch_device)
        if args.mode in ("long", "short12", "daily3y"):
            # Keep the last patch available even for a series with very short history.
            missing = (past < observed_start).reshape(-1, patch_count, tokens_per_patch).all(axis=-1)
            missing[:, -1] = False
            padding = torch.as_tensor(missing, device=torch_device)
        series = torch.as_tensor(series, device=torch_device)
        t = torch.as_tensor(t, device=torch_device)
        return model(cal[(t[:, None] + offsets).clamp_min(0)], cal[t], series, inputs, padding)

    @torch.no_grad()
    def predict(start,end):
        model.eval()
        forecasts = np.empty((len(ids),end-start), np.float32)
        for t in range(start,end):
            for i in range(0,len(ids),batch_size):
                series = np.arange(i,min(i+batch_size,len(ids)))
                forecasts[i:i+len(series),t-start] = pred(series,np.full_like(series,t)).cpu().numpy()*scale
        return forecasts

    def valid(start,end):
        return seen[:,None] & np.isfinite(raw[:,start:end])

    def wape(y,p,mask):
        return float(np.abs(y[mask]-p[mask]).sum(dtype=np.float64)/y[mask].sum(dtype=np.float64))

    if long:
        print(f"12 layers, 8 heads, {patch_count} tokens, batch {batch_size}", flush=True)
    best = float("inf")
    for epoch in range(1,3):
        model.train()
        for n,g in enumerate(random.permutation(len(groups)),1):
            series,times = groups[g]
            for step, part in enumerate(np.array_split(random.permutation(len(series)),(len(series)+batch_size-1)//batch_size), 1):
                si, ti = series[part], times[part]
                target = torch.as_tensor(raw[si,ti]/scale,device=torch_device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    yhat = pred(si,ti)
                loss = nn.functional.l1_loss(yhat.float(),target)
                loss.backward()
                opt.step()
                if long and n == 1 and step in (1, 10, 100):
                    print(f"epoch {epoch} step {step} loss {float(loss):.5f} peak_reserved_mib={torch.cuda.max_memory_reserved()/2**20:.0f}", flush=True)
            if n % (1 if long else 25) == 0:
                print(f"epoch {epoch}/2 block {n}/{len(groups)}",flush=True)
        score = wape(raw[:,fit_end:test],predict(fit_end,test),valid(fit_end,test))
        print(f"epoch {epoch} validation WAPE {score:.5f}",flush=True)
        if score < best:
            best, selected = score, epoch
            torch.save(model.state_dict(),f"outputs/{tag}.pt")
    model.load_state_dict(torch.load(f"outputs/{tag}.pt",weights_only=True))
    actual,p = raw[:,test:],predict(test,span)
    prior = raw[:,test-1:-1]
    mask = valid(test,span)
    ev = mask & np.isfinite(prior)
    # Accumulate exact frequencies of rounded, nonzero training changes.
    counts = {}
    for i in range(0,len(ids),rows_per_block):
        diff = np.abs(np.diff(raw[i:i+rows_per_block,:fit_end],axis=1))
        eligible = np.arange(1,fit_end)>first[i:i+rows_per_block,None]
        values, frequency = np.unique(diff[eligible & np.isfinite(diff) & (diff>0)].round(3),return_counts=True)
        for value,count in zip(values,frequency):
            counts[float(value)] = counts.get(float(value),0)+int(count)
    cutoff = 0.75 * sum(counts.values())
    cumulative = 0
    for floor,count in sorted(counts.items()):
        cumulative += count
        if cumulative >= cutoff:
            break
    times = np.arange(low+test,high+1)
    if date_based:
        times = times.astype("datetime64[D]")
    key, col = np.nonzero(mask)
    results = dict(dataset=args.dataset, mode=args.mode, weekly=weekly, raw_panel_rows=parquet.metadata.num_rows,
                   series=len(ids), seen_series=int(seen.sum()), rows=int(mask.sum()), parameters=sum(p.numel() for p in model.parameters()),
                   patch_days=tokens_per_patch, context_tokens=patch_count, train_windows=sum(len(x) for x,_ in groups), best_epoch=selected,
                   validation_wape=best, test_wape=wape(actual,p,mask), start=str(times[0]), end=str(times[-1]),
                   event_floor=floor, surge=event_metrics(actual[ev],p[ev],prior[ev],1,floor),
                   drop=event_metrics(actual[ev],p[ev],prior[ev],-1,floor))
    pd.DataFrame(dict(store=ids[key,0],product=ids[key,1],period=times[col],actual=actual[mask],
                      forecast=p[mask])).to_csv(f"outputs/{tag}-predictions.csv.gz",index=False)
    np.save(f"outputs/{tag}-series.npy",ids)
    fig, axes = plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    candidates = np.flatnonzero(np.nansum(actual,axis=1)>0)
    for ax,i in zip(axes.flat,random.choice(candidates,min(6,len(candidates)),replace=False)):
        ax.plot(times,actual[i],label="Actual")
        ax.plot(times,p[i],label="Forecast")
        ax.set_title(f"Store {ids[i,0]} / Product {ids[i,1]}")
        ax.tick_params(axis="x",rotation=25,labelsize=8)
    axes.flat[0].legend()
    fig.savefig(f"outputs/{tag}.png",dpi=150)
    Path(f"outputs/{tag}.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()










