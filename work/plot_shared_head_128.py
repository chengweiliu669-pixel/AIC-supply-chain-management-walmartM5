from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "outputs"
old = pd.read_csv(out / "m5_residual_comparison.csv", parse_dates=["date"])
previous = pd.read_csv(out / "m5_shared_head_daily_residuals.csv", parse_dates=["date"])
old = old.drop(columns=["forecast", "residual"]).merge(
    previous[["date", "forecast", "residual"]], on="date", validate="one_to_one"
)
new = pd.read_csv(out / "m5_shared_head_128_daily_residuals.csv", parse_dates=["date"])
assert len(old) == len(new) == 28
d = old.merge(new, on="date", suffixes=("_original", "_shared"), validate="one_to_one")
assert len(d) == 28 and d.actual_original.equals(d.actual_shared)
d = d.rename(columns={"actual_original": "actual"}).drop(columns=["actual_shared", "extra_absolute_error"])
d["absolute_error_improvement"] = d.residual_original.abs() - d.residual_shared.abs()
d.to_csv(out / "m5_shared_head_128_comparison.csv", index=False)

font = "C:/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                     "axes.unicode_minus": False, "font.size": 11})
fig, ax = plt.subplots(figsize=(13, 5.4), layout="constrained")
x = mdates.date2num(d.date)
focus = d.date.dt.strftime("%m-%d").eq("06-07")
ax.axvspan(x[focus][0] - .48, x[focus][-1] + .48, color="#fff0c7", zorder=0)
ax.bar(x - .27, d.residual_original, .25, color="#295bb0", label="共享头 · 32 步")
ax.bar(x, d.residual_shared, .25, color="#008372", label="共享头 · 128 步")
ax.bar(x + .27, d.weekly_naive_residual, .25, color="#8798a9", label="重复最近 7 天销量")
ax.axhline(0, color="#293549", lw=1)
ax.set_title("M5 · 共享预测头继续训练后的 28 天销量残差", loc="left", weight="bold", fontsize=18, pad=14)
ax.set_ylabel("预测 − 实际（件）")
ax.set_xlabel("日期（2016 年）")
ax.set_xticks(x)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.tick_params(axis="x", labelrotation=55, length=0)
for label, highlight in zip(ax.get_xticklabels(), focus):
    if highlight:
        label.set_weight("bold")
        label.set_color("#975500")
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e8f0")
ax.margins(y=.18)
ax.legend(loc="upper left", ncols=3, frameon=False)
fig.savefig(out / "m5_shared_head_128_comparison.png", dpi=160, facecolor="white")
print(d.loc[focus, ["date", "actual", "residual_original", "residual_shared", "weekly_naive_residual",
                    "absolute_error_improvement"]].to_string(index=False))
print("Daily aggregate MAE:", d[["residual_original", "residual_shared", "weekly_naive_residual"]].abs().mean().to_dict())

forecast = d.forecast_shared.to_numpy()
np.testing.assert_allclose(forecast[7:], forecast[:-7], atol=0.1)
print(pd.read_csv(out / "m5_shared_head_128_wrmsse.csv").tail(1).to_string(index=False))
print("Largest positive residual:", d.loc[d.residual_shared.idxmax(), ["date", "residual_shared"]].to_dict())
