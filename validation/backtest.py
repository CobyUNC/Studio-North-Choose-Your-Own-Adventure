"""Chronological validation of model variants against the sBG + Gamma reference.

    uv run --frozen python validation/backtest.py --data-dir /path/to/released/data

Only released observations are used: every origin truncates the tables to an
earlier cutoff, and outcomes come from later released months (through
December 2025). Reference predictions are produced separately by
``reference_backtest.py`` in the lab environment and read from
``validation/results/reference``.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import pandas as pd  # noqa: E402
import origins  # noqa: E402
import panel  # noqa: E402
import example_constant_hazard  # noqa: E402
from model import Config, FIRST_PASS, SELECTED, forecast, forecast_ensemble  # noqa: E402

# Every variant shares the panel, origins and score. Retention forms and
# research-driven variants are compared like-for-like; ablations remove one
# cash component at a time from the selected model.
SELECTED_RUN = lambda t, g: forecast_ensemble(t, g)
mod = lambda **kw: {k: Config(**{**v.__dict__, **kw}) for k, v in FIRST_PASS.items()}
VARIANTS = {
    'selected_ensemble': SELECTED_RUN,
    # First-pass selection and its members
    'first_pass_ensemble': FIRST_PASS,
    'sbg_cohort_trend': Config(hazard_kind='sbg', cohort_trend=True),
    'sbg_no_cohort_effect': Config(hazard_kind='sbg', cohort_trend=False),
    # Retention forms
    'sbg_cohort_random_walk': Config(hazard_kind='sbg', cohort_trend=False, cohort_rw=10.0),
    'bdw_cohort_trend': Config(hazard_kind='bdw', cohort_trend=True),
    'logit_hazard_cohort_ridge': Config(hazard_kind='logit', cohort_ridge=4.0),
    'example_constant_hazard': lambda t, g: example_constant_hazard.predict(t, g, seed=0),
    # Second-pass, research-driven variants (each applied to the first-pass ensemble)
    'r2_period_frozen': mod(period_effect=True),
    'r2_period_extrapolated': mod(period_effect=True, period_forecast='extrapolate'),
    'r2_voluntary_involuntary_split': mod(churn_event='split'),
    'r2_latent_class_heterogeneity': mod(heterogeneity='latent2'),
    'r2_cohort_partial_pooling': mod(cohort_ridge=4.0),
    'r2_recency_weight_halflife12': mod(trial_halflife=12.0),
    'r2_plus_bdw_member': {**FIRST_PASS, 'bdw': Config(hazard_kind='bdw', cohort_trend=True)},
    # Third pass: recent-level cash anchor (idea from a teammate's model), on the selected ensemble
    'r3_cash_anchor_prior100': {k: Config(**{**v.__dict__, 'cash_anchor': 100.0}) for k, v in SELECTED.items()},
    # Ablations of the selected ensemble's cash block
    'ablation_no_tier_mix': {k: Config(**{**v.__dict__, 'tier_aware': False}) for k, v in SELECTED.items()},
    'ablation_flat_costs': {k: Config(**{**v.__dict__, 'cost_trend': 'flat'}) for k, v in SELECTED.items()},
    'ablation_no_addon_trends': {k: Config(**{**v.__dict__, 'addon_calendar_trend': False,
                                              'addon_price_trend': False}) for k, v in SELECTED.items()},
}


def _predict(spec, tables, grid):
    if isinstance(spec, Config):
        return forecast(tables, grid, spec)
    if isinstance(spec, dict):
        return forecast_ensemble(tables, grid, spec)
    return spec(tables, grid)


def run(data_dir: Path, variants: dict, cutoffs=origins.CUTOFFS, save: Path | None = None):
    tables = origins.load_released(data_dir)
    full = panel.cohort_age_summary(tables, origins.RELEASE_CUTOFF)
    problems = {c: origins.origin_problem(tables, c, full) for c in cutoffs}
    rows = []
    for name, cfg in variants.items():
        for c, (past, grid, truth) in problems.items():
            t = time.monotonic()
            pred = _predict(cfg, past, grid)
            rows.append({'model': name, 'cutoff': c, **origins.score(pred, truth), 'seconds': time.monotonic() - t})
            if save is not None:
                (save / name).mkdir(parents=True, exist_ok=True)
                pred.to_csv(save / name / f'pred_{c}.csv', index=False, float_format='%.10f')
    ref_dir = HERE / 'results' / 'reference'
    for c, (_, grid, truth) in problems.items():
        f = ref_dir / f'pred_{c}.csv'
        if f.exists():
            rows.append({'model': 'reference_sbg_gamma', 'cutoff': c, **origins.score(pd.read_csv(f), truth)})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    sel = results[results.cutoff.isin(origins.SELECTION_CUTOFFS)]
    return (sel.groupby('model').agg(mean_wape=('contribution_wape', 'mean'),
                                      worst_wape=('contribution_wape', 'max'),
                                      mean_abs_bias=('portfolio_bias', lambda s: s.abs().mean()),
                                      mean_retention_wape=('retention_cell_wape', 'mean'),
                                      origins=('cutoff', 'size'))
            .sort_values('mean_wape'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE / 'results')
    a = p.parse_args()
    res = run(a.data_dir, VARIANTS, save=a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / 'backtest_by_origin.csv', index=False, float_format='%.6f')
    summ = summarize(res)
    summ.to_csv(a.out / 'backtest_summary.csv', float_format='%.6f')
    print(res.pivot(index='cutoff', columns='model', values='contribution_wape').round(4).to_string())
    print(summ.round(4).to_string())
