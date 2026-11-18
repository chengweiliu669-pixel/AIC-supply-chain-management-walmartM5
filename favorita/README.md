# Favorita daily sales forecasting

The selected completed full-panel experiment achieved **49.9963% test WAPE** using a fully trained Transformer with seven-day tokens and approximately three years of history.

[Start with the notebook](Favorita.ipynb). It includes optional validation inference, seasonal and trailing-mean baselines, bias and segment diagnostics, a chronological scale/blend experiment, and a prioritized plan for feature ablations. Analysis outputs are written under runs/favorita-improvement. The expensive analysis is disabled by default while other GPU jobs are running.

## Recorded results

| Metric | Value |
|---|---:|
| Validation WAPE | 49.1441% |
| Test WAPE | 49.9963% |
| Selected epoch | 1 of 2 |
| Scored daily observations | 4,843,748 |
| Item-store series | 174,685 |
| Trainable parameters | 7,240,641 |
| Training windows per epoch | 183,897,562 |
| Upward precision / recall | 63.19% / 17.95% |
| Downward precision / recall | 66.70% / 37.59% |

![Daily actual and predicted sales](forecast.png)

## Data and temporal evaluation

Download and extract train.csv from [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data), following the competition access terms.

The daily panel spans January 1, 2013 through August 15, 2017. Fit targets end June 20, 2017. Validation runs June 21–July 18 and test runs July 19–August 15. Lowest validation WAPE selects the checkpoint.

**Evaluation is rolling one-day-ahead.** A prediction for day t uses actual sales through t-1, including earlier actual observations inside the holdout. Weights stay fixed during evaluation. A fixed-origin 28-day forecast requires a separate experiment.

WAPE is sum(abs(actual - forecast)) / sum(actual), aggregated across scored item-store-day observations. Series first observed after the fit cutoff are excluded from scoring.

The panel aggregates unit_sales by store_nbr, item_nbr and date, sorted by those keys. Negative aggregate sales are clipped to zero by the training loader. Missing daily rows after the first observation become zero. Pre-launch history is masked. Target eligibility starts at max(28, first observed day + 1), so early samples have shorter observed histories.

## Inputs and model

The inputs are calendar encodings, learned item-store identity and historical sales with an observation indicator. Calendar inputs contain elapsed years and sine/cosine weekly, monthly and annual cycles. Promotion is retained in the prepared panel. The forward pass consumes calendar, identity and sales.

Each token encodes seven consecutive days. There are 157 tokens covering 1,099 days, with global attention across available history. A width-32 daily representation is concatenated within each patch and projected to width 128. The encoder has 12 layers, 8 heads, feed-forward width 256, dropout 0.05 and learned positions. A Softplus head combines the final historical token and target-date representation.

Training uses all model parameters, AdamW learning rate 0.001, batch size 1,024, BF16 autocast, FP32 L1 loss, seed 42 and two full epochs. All stores share one model.

Event labels compare daily actual and predicted sales with the previous actual day. The threshold is max(5 units, 50% of previous sales). The absolute floor was obtained from the 75th percentile of rounded, nonzero training changes. Upward TP/FP/FN are 67,883 / 39,541 / 310,370. Downward TP/FP/FN are 116,757 / 58,283 / 193,817.

## Reproduction

Run from the repository root using a CUDA-capable Python environment. Install a PyTorch build compatible with your GPU and driver, then install the remaining packages. The training host used an RTX 5080 Laptop 16GB. Its package inventory read after training was PyTorch 2.14.1+cu132, NumPy 2.5.4, pandas 3.1.1, PyArrow 26.0.1 and Matplotlib 3.11.2.

~~~sh
python -m pip install -r favorita/requirements.txt
python favorita/prepare.py --train data/favorita/train.csv
python favorita/train.py favorita long
~~~

Full training processes hundreds of millions of windows and can take multiple days. It writes outputs/favorita-patch-long.pt, JSON metrics, CSV.GZ daily predictions, a series mapping and a plot. A new run overwrites the same output names.

model.pt is the selected checkpoint. series.npy preserves its embedding order. metrics.json and forecast.png are recorded outputs. Input data and the full prediction table can be obtained or regenerated locally.

The published source was retrieved from the completed training host. Publication edits restrict the CLI to this experiment, extract its metric helper and ensure the output directory exists. Full retraining for this publication and bitwise equivalence across GPU/software versions are unverified.
