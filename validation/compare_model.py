"""Score any challenge submission on this repository's six-origin backtest.

Use this to compare another team's model with ours on identical, observed-only
history. The other model is run exactly as the instructor runs it, through its
own locked environment and standard command:

    uv run --frozen python main.py --data-dir DATA --prediction-grid GRID --output OUT --seed 20261116

For each historical origin this script writes the truncated eight CSVs and the
rebuilt grid to a temporary folder, runs the other repository's ``main.py``,
and scores the output against later *released* months (all before 2026). It
then prints a leaderboard next to our saved results.

    # 1. prepare the other team's environment once
    cd /path/to/other-team-repo && uv sync --frozen
    # 2. score it (from this repository)
    uv run --frozen python validation/compare_model.py \\
        --data-dir /path/to/released/data --repo /path/to/other-team-repo --name "Team X"

Running another team's code executes it on your machine; read it first.

A second mode compares two 2026 forecast files (no outcomes exist yet, so this
shows differences, not accuracy):

    uv run --frozen python validation/compare_model.py --compare-2026 \\
        predictions/selected_model_2026.csv /path/to/their/predictions.csv
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import pandas as pd  # noqa: E402
import origins  # noqa: E402
import panel  # noqa: E402

OURS = ['selected_ensemble', 'first_pass_ensemble', 'reference_sbg_gamma', 'example_constant_hazard']
LABELS = {'selected_ensemble': 'Studio North: Choose Your Own Adventure (selected)',
          'first_pass_ensemble': 'Our first pass',
          'reference_sbg_gamma': 'Released reference sBG + Gamma',
          'example_constant_hazard': 'Template constant-hazard example'}
PRED = ['predicted_retained', 'predicted_net_revenue', 'predicted_contribution']


def _write_inputs(tables: dict, folder: Path) -> None:
    folder.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f'{name}.csv', index=False)


def score_repo(data_dir: Path, repo: Path, seed: int, timeout: int) -> pd.DataFrame:
    tables = origins.load_released(data_dir)
    full = panel.cohort_age_summary(tables, origins.RELEASE_CUTOFF)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for cutoff in origins.CUTOFFS:
            past, grid, truth = origins.origin_problem(tables, cutoff, full)
            work = Path(tmp) / cutoff
            _write_inputs(past, work / 'data')
            grid.to_csv(work / 'grid.csv', index=False)
            out = work / 'out' / 'predictions.csv'
            cmd = ['uv', 'run', '--frozen', 'python', 'main.py', '--data-dir', str(work / 'data'),
                   '--prediction-grid', str(work / 'grid.csv'), '--output', str(out), '--seed', str(seed)]
            started = time.monotonic()
            try:
                run = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout)
                if run.returncode != 0:
                    raise RuntimeError(run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'nonzero exit')
                pred = pd.read_csv(out)
                rows.append({'cutoff': cutoff, **origins.score(pred, truth),
                             'seconds': time.monotonic() - started, 'status': 'scored'})
            except Exception as exc:  # recorded, never silently skipped
                rows.append({'cutoff': cutoff, 'status': f'failed: {exc}'})
            print(f'{cutoff}: {rows[-1].get("contribution_wape", rows[-1]["status"])}', flush=True)
    return pd.DataFrame(rows)


def leaderboard(name: str, theirs: pd.DataFrame) -> pd.DataFrame:
    saved = pd.read_csv(HERE / 'results' / 'backtest_by_origin.csv')
    saved = saved[saved.model.isin(OURS)].assign(model=lambda d: d.model.map(LABELS))
    both = pd.concat([saved, theirs.assign(model=name)], ignore_index=True)
    wide = both.pivot(index='model', columns='cutoff', values='contribution_wape')
    sel = both[both.cutoff.isin(origins.SELECTION_CUTOFFS)]
    board = pd.DataFrame({
        'mean_wape': wide[origins.SELECTION_CUTOFFS].mean(axis=1),
        'worst_wape': wide[origins.SELECTION_CUTOFFS].max(axis=1),
        'mean_abs_bias': sel.groupby('model').portfolio_bias.apply(lambda s: s.abs().mean()),
        'mean_retention_wape': sel.groupby('model').retention_cell_wape.mean()})
    board = board.join(wide).sort_values('mean_wape')
    return board


def compare_2026(ours: Path, theirs: Path) -> None:
    a, b = pd.read_csv(ours), pd.read_csv(theirs)
    j = a.merge(b, on=panel.KEYS, suffixes=('_a', '_b'), validate='one_to_one')
    if len(j) != len(a) or len(j) != len(b):
        raise ValueError('The two files do not cover the same grid')
    print(f'A = {ours}\nB = {theirs}\n')
    print('2026 totals')
    print(pd.DataFrame({'A': [a[c].sum() for c in PRED], 'B': [b[c].sum() for c in PRED]}, index=PRED)
          .assign(**{'B vs A': lambda d: d.B / d.A - 1}).round(3).to_string())
    by = j.groupby('cohort_month')[[c + '_a' for c in PRED[2:]] + [c + '_b' for c in PRED[2:]]].sum()
    by['B vs A'] = by.predicted_contribution_b / by.predicted_contribution_a - 1
    print('\n2026 contribution by acquisition cohort')
    print(by.round(3).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--repo', type=Path, help='Other team repository containing main.py and uv.lock')
    p.add_argument('--name', default='Other model')
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--timeout', type=int, default=3600, help='Seconds per origin (contract budget)')
    p.add_argument('--out', type=Path, help='Optional CSV for the leaderboard')
    p.add_argument('--compare-2026', nargs=2, type=Path, metavar=('OURS', 'THEIRS'))
    a = p.parse_args()
    if a.compare_2026:
        compare_2026(*a.compare_2026)
        sys.exit(0)
    if not (a.data_dir and a.repo):
        p.error('--data-dir and --repo are required for backtest scoring')
    theirs = score_repo(a.data_dir, a.repo.resolve(), a.seed, a.timeout)
    board = leaderboard(a.name, theirs[theirs.status.eq('scored')])
    pd.set_option('display.width', 200)
    print('\nCohort-month contribution WAPE (lower is better); mean/worst over the five selection origins')
    print((board * 100).round(2).to_string())
    if a.out:
        board.to_csv(a.out, float_format='%.6f')
