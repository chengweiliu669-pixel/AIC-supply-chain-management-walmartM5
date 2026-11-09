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
d = pd.read_csv(out / "m5_daily_residuals.csv", parse_dates=["date"])
last_week = pd.read_csv(root / "work/m5/sales_train_evaluation.csv",
                        usecols=[f"d_{i}" for i in range(1935, 1942)])
assert last_week.shape == (30490, 7) and len(d) == 28
d["weekly_naive_forecast"] = np.tile(last_week.sum().to_numpy(), 4)
d["weekly_naive_residual"] = d.weekly_naive_forecast - d.actual
d["extra_absolute_error"] = d.residual.abs() - d.weekly_naive_residual.abs()
worse = d.extra_absolute_error > 0
d.to_csv(out / "m5_residual_comparison.csv", index=False)

font = "C:/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                     "axes.unicode_minus": False, "font.size": 11})
fig, ax = plt.subplots(figsize=(13, 5.3), layout="constrained")
x = mdates.date2num(d.date)
ax.bar(x - .19, d.residual, .36, color="#295bb0", label="Time-LLM")
ax.bar(x + .19, d.weekly_naive_residual, .36, color="#8798a9", label="重复最近 7 天销量")
ax.axhline(0, color="#293549", lw=1)
ax.scatter(x[worse], np.full(worse.sum(), .025), marker="D", s=22, color="#b44e2e",
           transform=ax.get_xaxis_transform(), label="Time-LLM 绝对误差更大")
ax.set_ylim(min(d.residual.min(), d.weekly_naive_residual.min()) - 3500,
            max(d.residual.max(), d.weekly_naive_residual.max()) + 5500)
ax.set_title("M5 · 28 天销量预测残差对比", loc="left", weight="bold", fontsize=18, pad=14)
ax.set_ylabel("预测 − 实际（件）")
ax.set_xlabel("日期（2016 年）")
ax.set_xticks(x)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.tick_params(axis="x", labelrotation=55, length=0)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e8f0")
ax.legend(loc="upper left", ncols=3, frameon=False)
fig.savefig(out / "m5_residual_comparison.png", dpi=160, facecolor="white")
print("Worse", worse.sum(), d.loc[worse, "date"].dt.strftime("%m-%d").tolist())
print(d.nlargest(5, "extra_absolute_error")[["date", "residual", "weekly_naive_residual", "extra_absolute_error"]].to_string(index=False))
