"""Add high-error examples and full-panel diagnostic tables to the product gallery."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
args = parser.parse_args()
out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)
f = pd.read_csv(args.predictions, parse_dates=["period"])
f["residual"] = f.forecast - f.actual
f["error"] = f.residual.abs()
f["over_error"] = f.residual.clip(lower=0)
f["under_error"] = (-f.residual).clip(lower=0)
g = f.groupby(["store", "product"], sort=True).agg(
    sales=("actual", "sum"), forecast=("forecast", "sum"),
    error=("error", "sum"), mean=("actual", "mean"), std=("actual", "std"),
    maximum=("actual", "max"), days=("actual", "size"),
    nonzero=("actual", lambda x: (x > 0).sum()),
)
g["wape"] = g.error / g.sales.replace(0, np.nan)
g["bias"] = (g.forecast - g.sales) / g.sales.replace(0, np.nan)
g["cv"] = g["std"] / g["mean"].replace(0, np.nan)
low = (g.sales > 0) & (g.wape <= .3)
high = (g.sales > 0) & (g.wape > .3)
zero = g.sales == 0
rows = []
for name, mask in [("Low error (WAPE <= 30%)", low),
                   ("High error (WAPE > 30%)", high),
                   ("Zero actual sales", zero)]:
    x = g[mask]
    rows.append(dict(
        group=name, series=int(mask.sum()), share=float(mask.mean()),
        wape=float(x.error.sum()/x.sales.sum()) if x.sales.sum() else None,
        sales_share=float(x.sales.sum()/g.sales.sum()),
        error_share=float(x.error.sum()/g.error.sum()),
    ))
pd.DataFrame(rows).to_csv(out/"error-groups.csv", index=False)
# Every selected example has the complete 28-day test window.
pool = g[high & (g.days == 28) & (g["mean"] >= 5)].copy()
selected = []
used_stores, used_products = set(), set()
def select(label, candidates, order):
    for key, row in candidates.sort_values(order, kind="stable").iterrows():
        if key[0] not in used_stores and key[1] not in used_products:
            selected.append((label, key, row))
            used_stores.add(key[0])
            used_products.add(key[1])
            return
    raise RuntimeError(f"No example for {label}")

pool["median_distance"] = (pool.wape - g.loc[high, "wape"].median()).abs()
select("Near high-group median", pool[pool.nonzero >= 26], ["median_distance"])
select("Persistent underprediction", pool[(pool.bias < -.5) & (pool.nonzero >= 26)],
       ["error"])
select("Persistent overprediction", pool[(pool.bias > .5) & (pool.nonzero >= 26)],
       ["error"])
select("Intermittent sales and peaks", pool[(pool.nonzero >= 4) & (pool.nonzero <= 14)],
       ["wape"])

high_keys = pd.MultiIndex.from_frame(f[["store", "product"]]).isin(g.index[high])
h = f[high_keys].copy()
h_error = h.error.sum()
peak_threshold = f.groupby(["store","product"]).actual.transform(lambda x: x.quantile(.8))
peak_mask = high_keys & (f.actual >= peak_threshold) & (f.actual > 0)
stats = dict(
    total_series=len(g), total_rows=len(f), overall_wape=g.error.sum()/g.sales.sum(),
    high_median_wape=g.loc[high,"wape"].median(),
    high_p90_wape=g.loc[high,"wape"].quantile(.9),
    above_100pct=int((g.wape > 1).sum()),
    high_under_error_share=h.under_error.sum()/h_error,
    high_over_error_share=h.over_error.sum()/h_error,
    high_zero_day_error_share=h.loc[h.actual == 0,"error"].sum()/h_error,
    high_peak_day_error_share=f.loc[peak_mask,"error"].sum()/h_error,
    high_peak_day_row_share=float(peak_mask.sum()/high_keys.sum()),
)
(out/"error-diagnostics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
plt.rcParams.update({
    "font.family":"DejaVu Sans", "font.size":10,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.edgecolor":"#ccd3de", "xtick.color":"#667387", "ytick.color":"#667387",
    "figure.facecolor":"#f6f8fc", "axes.facecolor":"white",
    "savefig.facecolor":"#f6f8fc",
})
fig, axes = plt.subplots(2,2,figsize=(15,8.6))
fig.subplots_adjust(top=.83,bottom=.1,left=.07,right=.97,hspace=.63,wspace=.23)
fig.text(.07,.953,"Where daily forecasts miss",fontsize=23,fontweight="bold",color="#183b59")
fig.text(.07,.909,"Selected high-error examples · July 19–August 15, 2017",
         fontsize=12,color="#667387")
legend = [Line2D([0],[0],color="#183b59",lw=2,label="Actual sales"),
          Line2D([0],[0],color="#e58035",lw=2,ls="--",label="Transformer forecast")]
fig.legend(handles=legend,loc="upper right",bbox_to_anchor=(.97,.96),frameon=False)
export = []
selection_rows = []
for ax, (label,key,row) in zip(axes.flat,selected):
    data = f[(f.store == key[0]) & (f["product"] == key[1])].sort_values("period")
    ax.set_axisbelow(True)
    ax.grid(axis="y",color="#e6ebf2")
    ax.plot(data.period,data.actual,color="#183b59",lw=2,marker="o",ms=3)
    ax.plot(data.period,data.forecast,color="#e58035",lw=2,ls="--")
    ax.fill_between(data.period,data.actual,data.forecast,color="#e58035",alpha=.09)
    ax.set_title(f"{label}\nStore {key[0]} · Product {key[1]}",
                 loc="left",fontsize=11,fontweight="bold",color="#183b59",pad=14)
    ax.text(1,1.04,f"WAPE {row.wape:.1%}",transform=ax.transAxes,ha="right",color="#667387")
    ax.set_ylabel("Daily unit sales")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(length=0,pad=8)
    export.append(data[["store","product","period","actual","forecast"]])
    selection_rows.append(dict(pattern=label,store=int(key[0]),product=int(key[1]),
                               wape=row.wape,signed_bias=row.bias,mean_sales=row["mean"]))
fig.text(.07,.025,"Curated diagnostic cases. High-error group WAPE: 55.86%. Full-panel test WAPE: 49.9963%.",
         fontsize=10,color="#667387")
fig.savefig(out/"high-error-products.png",dpi=180)
plt.close(fig)
pd.concat(export).to_csv(out/"high-error-products.csv",index=False)
pd.DataFrame(selection_rows).to_csv(out/"high-error-selection.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(stats,indent=2))
print(pd.DataFrame(selection_rows).to_string(index=False))
