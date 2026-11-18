"""Select and render six low-error, non-flat held-out product examples."""
import argparse
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
destination = Path(__file__).resolve().parent / "figures"
destination.mkdir(exist_ok=True)
frame = pd.read_csv(args.predictions, parse_dates=["period"])
frame["absolute_error"] = (frame.actual-frame.forecast).abs()
stats = frame.groupby(["store","product"]).agg(
    days=("actual","size"), mean=("actual","mean"),
    std=("actual","std"), total=("actual","sum"),
    error=("absolute_error","sum"), nonzero=("actual",lambda x: (x>0).sum())
)
stats["wape"] = stats.error/stats.total.replace(0,np.nan)
stats["cv"] = stats["std"]/stats["mean"]
pool = stats[(stats.days==28)&(stats["mean"]>=10)&(stats.nonzero>=26)&
             (stats.cv>=.2)&(stats.wape<=.3)].copy()
# Favor visual clarity and low error while retaining different sales levels.
selections = []
used_stores = set()
used_products = set()
for lower,upper in [(100,np.inf),(60,100),(30,60),(15,30),(10,15),(10,np.inf)]:
    candidates = pool[(pool["mean"]>=lower)&(pool["mean"]<upper)]
    candidates = candidates.sort_values(["wape","mean"],ascending=[True,False])
    for key,row in candidates.iterrows():
        if key[0] not in used_stores and key[1] not in used_products:
            selections.append((key,row))
            used_stores.add(key[0])
            used_products.add(key[1])
            break
if len(selections)!=6:
    raise RuntimeError("Selection criteria did not yield six distinct examples")
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#ccd3de","axes.labelcolor":"#465265",
    "xtick.color":"#667387","ytick.color":"#667387",
    "figure.facecolor":"#f6f8fc","axes.facecolor":"white",
    "savefig.facecolor":"#f6f8fc",
})
actual_color="#183b59"
forecast_color="#e58035"
dates = pd.date_range("2017-07-19",periods=28)
chosen=[]
def draw(ax, data, key, score):
    ax.set_axisbelow(True)
    ax.grid(axis="y",color="#e6ebf2",linewidth=.8)
    ax.plot(data.period,data.actual,color=actual_color,linewidth=2.0,
            marker="o",markersize=3,markeredgewidth=0)
    ax.plot(data.period,data.forecast,color=forecast_color,linewidth=2,
            linestyle=(0,(4,2)))
    ax.fill_between(data.period,data.actual,data.forecast,color=forecast_color,alpha=.09)
    ax.set_title(f"Store {key[0]} · Product {key[1]}",loc="left",
                 fontsize=12,fontweight="bold",color="#183b59",pad=14)
    ax.text(1,1.05,f"WAPE {score:.1%}",transform=ax.transAxes,
            ha="right",color="#68778a",fontsize=10)
    ax.set_ylabel("Daily unit sales")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(length=0,pad=8)
    ax.margins(x=.02)
legend=[Line2D([0],[0],color=actual_color,lw=2,label="Actual sales"),
        Line2D([0],[0],color=forecast_color,lw=2,ls="--",label="Transformer forecast")]
fig,axes=plt.subplots(3,2,figsize=(15,11))
fig.subplots_adjust(top=.86,bottom=.07,left=.07,right=.97,hspace=.53,wspace=.23)
fig.text(.07,.962,"Actual vs predicted daily sales",fontsize=25,fontweight="bold",color=actual_color)
fig.text(.07,.925,"Favorita · July 19–August 15, 2017 · Seven-day tokens / three-year history",
         fontsize=12,color="#667387")
fig.legend(handles=legend,loc="upper right",bbox_to_anchor=(.97,.965),frameon=False)
fig.text(.07,.015,"Curated low-error, non-flat examples. Full-panel test WAPE: 49.9963%. Rolling one-day-ahead evaluation.",
         fontsize=10,color="#667387")
for ax,(key,row) in zip(axes.flat,selections):
    data=frame[(frame.store==key[0])&(frame["product"]==key[1])].sort_values("period")
    draw(ax,data,key,row.wape)
    chosen.append(data)
    single,sax=plt.subplots(figsize=(10,4.8))
    single.subplots_adjust(top=.78,bottom=.18,left=.1,right=.96)
    draw(sax,data,key,row.wape)
    single.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
    single.text(.1,.035,"Selected example · Full-panel WAPE 49.9963% · Rolling daily forecast",
                fontsize=9,color="#667387")
    single.savefig(destination/f"store-{key[0]}-product-{key[1]}.png",dpi=180)
    plt.close(single)
fig.savefig(destination/"selected-products.png",dpi=180)
plt.close(fig)
pd.concat(chosen).drop(columns="absolute_error").to_csv(destination/"selected-products.csv",index=False)
pd.DataFrame([dict(store=k[0],product=k[1],wape=r.wape,mean=r["mean"],cv=r.cv)
              for k,r in selections]).to_csv(destination/"selection-metrics.csv",index=False)
print(pd.read_csv(destination/"selection-metrics.csv").to_string(index=False))

