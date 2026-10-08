# Assignment 1: Polynomial Regression

Rupavatia Ansh R - IMT2024057

Polynomial regression for two datasets. The selected models are degree-5 Lasso
with alpha 0.01 for var1 and degree-10 Ridge with alpha 1 for var2.

## Setup

Tested with Python 3.12.

```text
python -m pip install -r requirements.txt
```

Place these files in a local `data/` folder. Do not upload the datasets to GitHub.

```text
IMT2024057_train_var1.csv
IMT2024057_test_var1.csv
IMT2024057_train_var2.csv
IMT2024057_test_var2.csv
```

## Training

```text
python train.py data results 1
python train.py data results 2
```

The scripts create `results/` and save predictions, fitted models, CV results
and evaluation summaries there. The full search took about 25 minutes on the
machine used for this assignment; runtime depends on the computer.

An 80/20 grouped split reserves the hold-out before tuning. Five-fold grouped CV
on the development data selects the model family, degree and alpha. Scaling is
fitted within each fold. Ridge and Lasso are compared at every allowed degree;
OLS is a baseline at degrees 1-3. The selected configuration is evaluated on the
hold-out, then refitted on all training rows.

## Inference

After training, predict from the saved models without repeating the search:

```text
python inference.py results/model_var1.joblib data/IMT2024057_test_var1.csv results/IMT2024057_pred_var1.csv
python inference.py results/model_var2.joblib data/IMT2024057_test_var2.csv results/IMT2024057_pred_var2.csv
```

Each prediction CSV has one `y` column, no index, and the original test-row order.

## Report figures

```text
python plots.py results
```

This recreates the degree-comparison and hold-out plots in
`results/report_figures/`. The PDF report and prediction CSVs are submitted
separately from this repository.
