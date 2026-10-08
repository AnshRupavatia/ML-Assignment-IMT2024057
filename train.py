"""Train polynomial regression models for var1 or var2."""
import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

ROLL = 'IMT2024057'
RIDGE_ALPHAS = [0.0001, 0.01, 0.1, 1, 10, 100, 1000]
LASSO_ALPHAS = {1: [0.05, 0.02, 0.01, 0.005, 0.002],
                2: [0.03, 0.01, 0.003, 0.001, 0.0005, 0.0002]}
MAX_ITER, TOL = 100000, 1e-5


def estimator(family, alpha, warm_start=False):
    if family == 'OLS':
        return LinearRegression()
    if family == 'Ridge':
        return Ridge(alpha=alpha, solver='svd')
    return Lasso(alpha=alpha, max_iter=MAX_ITER, tol=TOL,
                 warm_start=warm_start, selection='cyclic')


def fit_checked(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return False
    for warning in caught:
        warnings.warn(str(warning.message), warning.category)
    return True


def load_data(data_dir, variant):
    features = [f'x{i}' for i in range(1, (6 if variant == 1 else 3) + 1)]
    train = pd.read_csv(data_dir / f'{ROLL}_train_var{variant}.csv')
    test = pd.read_csv(data_dir / f'{ROLL}_test_var{variant}.csv')
    if list(train.columns) != features + ['y'] or list(test.columns) != features:
        raise ValueError('Unexpected input columns or column order.')
    if not all(np.isfinite(frame.to_numpy(dtype=float)).all() for frame in [train, test]):
        raise ValueError('Inputs must contain only finite numeric values.')
    return train, test, features


def select_model(X, y, groups, variant, out):
    folds = list(GroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
    rows = []
    for degree in range(1, (10 if variant == 1 else 20) + 1):
        # Polynomial expansion learns no statistics; scaling is fitted per fold.
        expanded = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
        candidates = [('Ridge', a) for a in RIDGE_ALPHAS]
        candidates += [('Lasso', a) for a in LASSO_ALPHAS[variant]]
        if degree <= 3:
            candidates += [('OLS', 0.0)]
        predictions = {candidate: np.full(len(y), np.nan) for candidate in candidates}
        failures = {candidate: 0 for candidate in candidates}
        for fold_number, (fitting, validation) in enumerate(folds, start=1):
            print(f'var{variant}: degree {degree}, fold {fold_number}/5', flush=True)
            scaler = StandardScaler().fit(expanded[fitting])
            A, B = scaler.transform(expanded[fitting]), scaler.transform(expanded[validation])
            # Reuse coefficients along the alpha path, checking convergence each time.
            lasso = estimator('Lasso', LASSO_ALPHAS[variant][0], warm_start=True)
            for family, alpha in candidates:
                model = lasso.set_params(alpha=alpha) if family == 'Lasso' else estimator(family, alpha)
                if fit_checked(model, A, y[fitting]):
                    predictions[(family, alpha)][validation] = model.predict(B)
                else:
                    failures[(family, alpha)] += 1
                    print(f'  Excluding {family} alpha={alpha:g}: convergence limit reached.', flush=True)
        for family, alpha in candidates:
            prediction = predictions[(family, alpha)]
            valid = failures[(family, alpha)] == 0
            rows.append(dict(degree=degree, family=family, alpha=alpha,
                             n_terms=expanded.shape[1], failed_folds=failures[(family, alpha)],
                             cv_mse=mean_squared_error(y, prediction) if valid else np.nan,
                             cv_r2=r2_score(y, prediction) if valid else np.nan))
        table = pd.DataFrame(rows)
        table.to_csv(out / f'cv_results_var{variant}.csv', index=False)
        best = table.dropna(subset=['cv_mse']).sort_values(['cv_mse', 'degree', 'alpha']).iloc[0]
        print(f'var{variant} degree {degree}: best so far {best.family}, d={best.degree}, alpha={best.alpha:g}, MSE={best.cv_mse:.6f}', flush=True)
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('variant', type=int, choices=[1, 2])
    args = parser.parse_args()
    out, variant = args.out_dir, args.variant
    out.mkdir(parents=True, exist_ok=True)
    threadpool_limits(1)
    train, test, features = load_data(args.data_dir, variant)
    X, y = train[features].to_numpy(), train.y.to_numpy()
    groups = pd.MultiIndex.from_frame(train[features]).factorize()[0]
    development, holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7).split(X, y, groups))
    assert not set(groups[development]) & set(groups[holdout])
    best, table = select_model(X[development], y[development], groups[development], variant, out)
    degree, alpha, family = int(best.degree), float(best.alpha), str(best.family)

    def pipeline():
        return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                             StandardScaler(), estimator(family, alpha))

    assessment = pipeline()
    if not fit_checked(assessment, X[development], y[development]):
        raise RuntimeError('Selected development model did not converge; no predictions written.')
    hp = assessment.predict(X[holdout])
    pd.DataFrame({'row_index': holdout, 'actual': y[holdout], 'prediction': hp}).to_csv(
        out / f'holdout_predictions_var{variant}.csv', index=False)
    # Refit the selected configuration on all labels after the hold-out assessment.
    final = pipeline()
    if not fit_checked(final, X, y):
        raise RuntimeError('Final model did not converge; no predictions written.')
    prediction = final.predict(test[features].to_numpy())
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError('Invalid predictions.')
    pd.DataFrame({'y': prediction}).to_csv(out / f'{ROLL}_pred_var{variant}.csv', index=False)
    joblib.dump({'model': final, 'features': features, 'roll': ROLL, 'variant': variant},
                out / f'model_var{variant}.joblib')
    fp = final.predict(X)
    summary = dict(variant=variant, family=family, degree=degree, alpha=alpha,
                   n_train=len(train), n_test=len(test), n_development=len(development), n_holdout=len(holdout),
                   n_unique_inputs=len(np.unique(groups)), n_terms=int(best.n_terms),
                   n_nonzero=int(np.count_nonzero(np.abs(final[-1].coef_) > 1e-8)),
                   cv_mse=float(best.cv_mse), cv_r2=float(best.cv_r2),
                   holdout_mse=mean_squared_error(y[holdout], hp), holdout_r2=r2_score(y[holdout], hp),
                   train_mse=mean_squared_error(y, fp), train_r2=r2_score(y, fp),
                   n_candidates=len(table), excluded_candidates=int(table.cv_mse.isna().sum()),
                   lasso_tolerance=TOL, lasso_max_iter=MAX_ITER,
                   split_seed=7, cv_seed=42, development_indices=development.tolist(), holdout_indices=holdout.tolist())
    (out / f'summary_var{variant}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('_indices')}, indent=2))


if __name__ == '__main__':
    main()
