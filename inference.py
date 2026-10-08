"""Predict from a saved pipeline without retraining."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('test_csv', type=Path)
    parser.add_argument('output_csv', type=Path)
    args = parser.parse_args()
    threadpool_limits(1)
    saved = joblib.load(args.model)  # Load only your own trusted model files.
    test = pd.read_csv(args.test_csv)
    if list(test.columns) != saved['features'] or not np.isfinite(test.to_numpy(dtype=float)).all():
        raise ValueError('Test input schema or values are invalid.')
    prediction = saved['model'].predict(test.to_numpy())
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite predictions.')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y': prediction}).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
