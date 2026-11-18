"""Shared chronological-validation design and score (no 2026 outcomes exist here).

Each historical origin is the last fully observed month before an exclusive
cutoff. Tables are truncated to what that release would have contained, the
forecast grid is rebuilt from the truncated history, and the realized outcomes
come from later *released* months (all before 2026-01-01).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import panel  # noqa: E402

RELEASE_CUTOFF = '2026-01-01'
# Exclusive cutoffs; horizons are truncated at December 2025.
CUTOFFS = ['2024-07-01', '2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01']
# Model selection uses origins with at least six future months.
SELECTION_CUTOFFS = ['2024-07-01', '2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01']


def load_released(data_dir: Path) -> dict[str, pd.DataFrame]:
    contract = json.loads((ROOT / 'submission-contract.json').read_text())
    tables = {}
    for name, fields in contract['input_tables'].items():
        dtypes = {f: ('string' if k in {'VARCHAR', 'DATE', 'TIMESTAMP'} else
                      'Int64' if k == 'INTEGER' else 'float64') for f, k in fields.items()}
        tables[name] = pd.read_csv(Path(data_dir) / f'{name}.csv', dtype=dtypes)
    return tables


def origin_problem(tables: dict, cutoff: str, full_summary: pd.DataFrame):
    """(truncated tables, grid, truth) for one historical origin."""
    past = panel.truncate_tables(tables, cutoff)
    summary = panel.cohort_age_summary(past, cutoff)
    origin = summary.month.max()
    horizon = min(12, int(full_summary.month.max() - origin))
    grid = panel.prediction_grid(summary, horizon)
    truth = grid.merge(full_summary[panel.KEYS + ['n_retained', 'net_revenue', 'contribution']],
                       on=panel.KEYS, how='left', validate='one_to_one')
    if truth[['n_retained', 'contribution']].isna().any().any():
        raise ValueError('Missing realized outcome for a validation key')
    return past, grid, truth


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """The contract's primary score and secondary diagnostics."""
    j = truth.merge(pred[panel.KEYS + ['predicted_retained', 'predicted_net_revenue',
                                       'predicted_contribution']], on=panel.KEYS, validate='one_to_one')
    cm = j.groupby(['cohort_month', 'calendar_month'])[['predicted_contribution', 'contribution']].sum()
    denom = float(cm.contribution.abs().sum())
    diff = cm.predicted_contribution - cm.contribution
    return {'contribution_wape': float(diff.abs().sum() / denom),
            'retention_cell_wape': float((j.predicted_retained - j.n_retained).abs().sum() / j.n_retained.sum()),
            'portfolio_bias': float(diff.sum() / denom),
            'points': int(len(cm)), 'rows': int(len(j))}
