from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, ticker
import pandas as pd

out = Path(__file__).resolve().parents[1] / "outputs"
d = pd.read_csv(out / "m5_shifted_origin_residuals.csv", parse_dates=["date"])
assert set(d.origin_shift) == {-1, 0, 1, 2}
assert (d.groupby("date").actual.nunique() == 1).all()
font = "C:/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                     "axes.unicode_minus": False, "font.size": 11})
fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
focus = pd.Timestamp("2016-06-07")
ax.axvline(focus, ls="--", color="#333b4e", alpha=.75, label="6 月 7 日")
styles = {-1: ("前移 1 天", "#008378"), 0: ("原起点", "#294ab0"),
          1: ("后移 1 天", "#b74729"), 2: ("后移 2 天", "#8b4e9f")}
for shift, (label, color) in styles.items():
    s = d[d.origin_shift.eq(shift)]
    ax.plot(s.date, s.residual, color=color, lw=2.1, alpha=.9, label=label)
    h = s[s.horizon.eq(16)]
    ax.scatter(h.date, h.residual, s=85, marker="D", color=color, edgecolor="white", zorder=5)
ax.scatter([], [], color="#333b4e", marker="D", s=60, label="第 16 个输出")
ax.axhline(0, color="#7c8998", lw=1)
ax.set_title("固定 Time-LLM 权重 · 移动预测起点后的每日残差", loc="left", fontsize=17, weight="bold")
ax.set_ylabel("预测 − 实际（件）")
ax.set_xlabel("日期（2016 年）")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax.grid(axis="y", color="#dfe6ee")
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=6, frameon=False)
fig.savefig(out / "m5_shifted_origin_residuals.png", dpi=165, facecolor="white")
print(d[d.date.eq(focus) | d.horizon.eq(16)].round(1).to_string(index=False))
print("Maxima per origin")
print(d.loc[d.groupby("origin_shift").residual.idxmax()].round(1).to_string(index=False))
