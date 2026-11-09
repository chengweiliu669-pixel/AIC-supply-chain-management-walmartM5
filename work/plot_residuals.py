from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parents[1] / "outputs"
daily = pd.read_csv(out / "m5_daily_residuals.csv", parse_dates=["date"])
detail = pd.read_csv(out / "m5_item_store_residuals.csv", parse_dates=["date"])
assert len(daily) == 28 and len(detail) == 30490 * 28
assert daily.date.is_unique and daily.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
np.testing.assert_allclose(daily.forecast - daily.actual, daily.residual)
np.testing.assert_allclose(detail.forecast - detail.actual, detail.residual, atol=2e-6)
totals = detail.groupby("date")[["actual", "forecast", "residual"]].sum()
np.testing.assert_allclose(totals, daily.set_index("date")[totals.columns], atol=0.02)

font_path = Path("C:/Windows/Fonts/msyh.ttc")
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False})
fig, (top, ax) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.4),
                              height_ratios=(1, 1.18), layout="constrained")
fig.suptitle("Time-LLM · M5 的 28 天销量预测残差", fontsize=19, weight="bold")
top.set_title("全部商品与门店的每日合计", loc="left", fontsize=12)
top.plot(daily.date, daily.actual, color="#21304a", lw=2.2, label="实际销量")
top.plot(daily.date, daily.forecast, color="#627ad2", lw=2.1, ls="--", label="预测销量")
top.legend(frameon=False, loc="upper right", ncols=2)
top.set_ylabel("销量（件）")
ax.set_title("每日残差 = 预测 − 实际", loc="left", fontsize=12)
ax.bar(daily.date, daily.residual, width=0.73,
       color=np.where(daily.residual >= 0, "#247c83", "#c45639"))
ax.axhline(0, color="#24354b", linewidth=1)
ax.set_ylabel("残差（件）")
ax.set_xlabel("日期（2016 年）")
ax.set_xticks(daily.date.iloc[[0, 4, 8, 12, 16, 20, 24, 27]])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
for panel in (top, ax):
    panel.spines[["top", "right", "left"]].set_visible(False)
    panel.spines["bottom"].set_color("#d5dde7")
    panel.grid(axis="y", color="#e2e8f0")
    panel.set_axisbelow(True)
    panel.tick_params(length=0, pad=6)
    panel.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
fig.savefig(out / "m5_residuals.png", dpi=170, facecolor="white")
print(daily.round(2).to_string(index=False))
print("Total:", daily[["actual", "forecast", "residual"]].sum().to_dict())
