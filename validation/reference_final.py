"""Reference sBG + Gamma forecast on the official 2026 grid (for comparison only).

    cd customer-economics-lab
    uv run --frozen python ../<team-repo>/validation/reference_final.py --data-dir data --grid ../<team-repo>/prediction_grid.csv --out ../<team-repo>/validation/results/reference/final_2026.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from cohort_lab.model import _posterior, _future, _economic_ratios, _validate  # noqa: E402
import origins  # noqa: E402
import panel  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--grid', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--seed', type=int, default=20261116)
a = p.parse_args()
frame = _validate(panel.cohort_age_summary(origins.load_released(a.data_dir), origins.RELEASE_CUTOFF))
grid = pd.read_csv(a.grid)
draws, _ = _posterior(frame, a.seed, 3000)
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
pd.concat(rows).to_csv(a.out, index=False, float_format='%.10f')
