"""Generate report figures from recorded CV and untouched hold-out results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', type=Path, nargs='?', default=Path('.'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.results_dir
    out = args.output_dir or root / 'report_figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    colors = {'Ridge': '#2166ac', 'Lasso': '#1b7837', 'OLS': '#b35806'}
    for v in [1, 2]:
        s = json.loads((root / f'summary_var{v}.json').read_text())
        table = pd.read_csv(root / f'cv_results_var{v}.csv').dropna(subset=['cv_mse'])
        fig, ax = plt.subplots(figsize=(6.2, 2.65), layout='constrained')
        for family in ['OLS', 'Ridge', 'Lasso']:
            scores = table[table.family == family].groupby('degree').cv_mse.min()
            ax.plot(scores.index, scores.values, '-o', markersize=3, linewidth=1.3,
                    color=colors[family], label=family if family != 'OLS' else 'OLS (degrees 1–3)')
        ax.axvline(s['degree'], linestyle='--', color='#555555', linewidth=1,
                   label=f"Selected degree {s['degree']}")
        ax.set_yscale('log')
        ax.set(xlabel='Total polynomial degree', ylabel='Development CV MSE (log scale)',
               xlim=(1, 10 if v == 1 else 20))
        ax.set_xticks(range(1, 11) if v == 1 else [1, 5, 10, 15, 20])
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc='upper right')
        fig.savefig(out / f'var{v}_degree_comparison.pdf')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.1), layout='constrained')
    for ax, v in zip(axes, [1, 2]):
        holdout = pd.read_csv(root / f'holdout_predictions_var{v}.csv')
        s = json.loads((root / f'summary_var{v}.json').read_text())
        ax.scatter(holdout.actual, holdout.prediction, s=11, alpha=.6, color=colors[s['family']],
                   edgecolors='none', rasterized=False)
        lo = min(holdout.actual.min(), holdout.prediction.min())
        hi = max(holdout.actual.max(), holdout.prediction.max())
        ax.plot([lo, hi], [lo, hi], '--', color='#555555', linewidth=1)
        ax.set(xlabel='Actual target', ylabel='Prediction', title=f'var{v}: untouched hold-out')
        ax.text(.04, .96, f"MSE = {s['holdout_mse']:.4f}\n$R^2$ = {s['holdout_r2']:.4f}",
                transform=ax.transAxes, va='top', fontsize=8,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    fig.savefig(out / 'holdout_predictions.pdf')
    plt.close(fig)
    print(f'Created three vector PDF figures in {out}')


if __name__ == '__main__':
    main()
