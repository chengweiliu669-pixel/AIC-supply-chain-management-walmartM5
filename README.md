# Walmart M5 Forecasting with Time-LLM

Local CPU experiments on the 28-day M5 Accuracy private evaluation window (May 23–June 19, 2016). Models forecast 30,490 item–store series. WRMSSE uses the published weights and all 12 M5 aggregation levels. Daily residuals in the figures sum all item–store predictions and subtract actual unit sales.

| Method | WRMSSE | June 7 residual (units) |
| --- | ---: | ---: |
| Repeat last 7 days | 0.8470 | −1,861 |
| Original head | 1.2915 | +22,357 |
| Original head with calendar adjustment | 1.2719 | +23,379 |
| Shared weekly head | 1.1776 | +5,793 |

![28-day residual comparison](outputs/m5_shared_head_residual_comparison.png)

The fixed-weight [origin-shift experiment](outputs/m5_shifted_origin_residuals.png) moved the largest positive residual with output position 16 across all four origins. The shared head uses a fixed seven-day basis, so predictions repeat by week. The original and modified runs use 56 input days, a frozen one-layer GPT-2 backbone and 32 training batches.

## Reproduce locally

Use Python 3.11, Git and PowerShell. Run from the repository root.

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements.txt

git clone https://github.com/KimMeen/Time-LLM.git work/Time-LLM
git -C work/Time-LLM checkout b13e881f86cd0475ce1b72c17110430663334955
git -C work/Time-LLM apply ../../time-llm-cpu.patch

curl.exe -fL https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip -o work/m5.zip
Expand-Archive work/m5.zip work/m5
$env:HF_HOME = "$pwd/work/hf"

& $python work/run_m5.py
& $python work/plot_residuals.py
& $python work/compare_residuals.py
& $python work/run_m5_calendar.py
& $python work/plot_calendar_comparison.py
& $python work/run_m5_shift.py
& $python work/plot_shift.py
& $python work/run_m5_shared_head.py
& $python work/plot_shared_head.py
```

The first run downloads pretrained GPT-2 weights from Hugging Face. The plotting scripts use Windows Microsoft YaHei for Chinese labels. Raw competition files, downloaded weights and per-item output files are kept locally.

Sources: [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy), [organizer data and weights](https://github.com/Mcompetitions/M5-methods), [Time-LLM upstream](https://github.com/KimMeen/Time-LLM).
