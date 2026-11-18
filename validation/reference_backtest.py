"""Refit the released hierarchical sBG + Gamma reference at each historical origin.

Runs in the *lab* environment because it imports the unchanged reference code:

    cd customer-economics-lab
    uv run --frozen python ../<team-repo>/validation/reference_backtest.py --data-dir data --out ../<team-repo>/validation/results/reference

The reference functions (`_posterior`, `_future`, `_economic_ratios`) are used
exactly as released; only the input history is truncated at each origin.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from cohort_lab.model import _posterior, _future, _economic_ratios, _validate  # noqa: E402
import origins  # noqa: E402
import panel  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--cutoffs', nargs='*', default=origins.CUTOFFS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tables = origins.load_released(a.data_dir)
    full = panel.cohort_age_summary(tables, origins.RELEASE_CUTOFF)
    for cutoff in a.cutoffs:
        target = a.out / f'pred_{cutoff}.csv'
        if target.exists():
            continue
        past, grid, truth = origins.origin_problem(tables, cutoff, full)
        frame = _validate(panel.cohort_age_summary(past, cutoff))
        started = time.monotonic()
        draws, diagnostic = _posterior(frame, a.seed, a.steps)
        rows = []
        for key, g in frame.groupby(panel.CELL, sort=True):
            last = g.iloc[-1]
            fut = grid[(grid.cohort_month == key[0]) & (grid.channel == key[1]) & (grid.primary_need == key[2])].copy()
            ages = fut.age_month.astype(int).tolist()
            retained, gross = _future(draws, last, ages, float(last.n_retained))
            net_ratio, contrib_ratio = _economic_ratios(frame, last, ages)
            fut['predicted_retained'] = retained.mean(axis=0)
            fut['predicted_net_revenue'] = (gross * net_ratio).mean(axis=0)
            fut['predicted_contribution'] = (gross * contrib_ratio).mean(axis=0)
            rows.append(fut)
        pred = pd.concat(rows, ignore_index=True)
        pred.to_csv(target, index=False, float_format='%.10f')
        s = origins.score(pred, truth) | {'cutoff': cutoff, 'seconds': time.monotonic() - started}
        print(json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
