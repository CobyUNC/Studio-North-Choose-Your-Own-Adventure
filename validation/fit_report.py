"""Fitted settings and forecast diagnostics at the December 2025 origin.

    uv run --frozen python validation/fit_report.py --data-dir /path/to/released/data

Writes validation/results/final_fit.json. Uses only the released tables.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import model  # noqa: E402
import origins  # noqa: E402
import panel  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data-dir', type=Path, required=True)
a = p.parse_args()
tables = origins.load_released(a.data_dir)
grid = pd.read_csv(HERE.parent / 'prediction_grid.csv')
origin, acq, sp, tx = model._history(tables, origins.RELEASE_CUTOFF)
report = {'origin_month': panel.index_to_month([origin])[0], 'members': {}}
cells = pd.DataFrame([(c, n) for c in ['search', 'partner'] for n in ['save_time', 'build_skill']],
                     columns=['channel', 'primary_need'])
for name, cfg in model.SELECTED.items():
    hz = model.MixtureHazard(cfg, origin).fit(sp)
    entry = {'converged': hz.converged, 'cells': []}
    for r in cells.itertuples(index=False):
        for cohort in ['2024-01-01', '2025-12-01']:
            ci = int(panel.month_index([cohort])[0])
            f = pd.DataFrame({'channel': r.channel, 'primary_need': r.primary_need, 'cohort_index': ci,
                              'age': [1, 6, 12, 23, 24, 30, 35], 'month': 0})
            xm, xc, _ = hz._designs(f)
            km = xm.shape[1]
            from scipy.special import expit
            mu = float(expit(xm[0] @ hz.params[:km])); phi = float(np.exp(xc[0] @ hz.params[km:km + xc.shape[1]]))
            entry['cells'].append({'channel': r.channel, 'primary_need': r.primary_need, 'cohort': cohort,
                                   'alpha': mu * phi, 'beta': (1 - mu) * phi,
                                   'monthly_churn_at_age': dict(zip(map(str, f.age), map(float, hz.predict(f))))})
    report['members'][name] = entry
cash = model.Cash(model.SELECTED['sbg_cohort_trend'], origin).fit(sp, tx)
months = np.arange(origin + 1, origin + 13)
report['cash'] = {
    'tier_prices': {('premium' if k else 'standard'): v for k, v in cash.price.items()},
    'refund_share_of_gross': cash.refund_share,
    'addon_cost_per_unit': cash.addon_cost,
    'addon_price_per_unit_jan_dec_2026': [float(x) for x in cash.addon_unit_price(months)[[0, -1]]],
    'standard_cost_jan_dec_2026': [float(x) for x in cash.sub_cost(False, months)[[0, -1]]],
    'premium_cost_jan_dec_2026': [float(x) for x in cash.sub_cost(True, months)[[0, -1]]],
    'standard_cost_last_observed': float(cash.cost[False][1][-1]),
    'premium_cost_last_observed': float(cash.cost[True][1][-1]),
    'monthly_upgrade_hazard_age_6_24': [float(x) for x in cash.upgrade(pd.DataFrame(
        {'channel': 'search', 'primary_need': 'save_time', 'age': [6, 24]}))],
    'downgrade_rate': cash.downgrade_rate,
    'addon_beta[intercept,build,partner,premium,log_age,calendar_per_year]': [float(x) for x in cash.addon_beta],
}
survivors = sp[(sp.month == origin) & sp.paid]
report['origin_survivors'] = int(len(survivors))
report['origin_premium_share'] = float(survivors.premium.mean())
pred = model.forecast_ensemble(tables, grid)
ref_path = HERE / 'results' / 'reference' / 'final_2026.csv'
tot = lambda d: {k: float(d[k].sum()) for k in model.PREDICTIONS}
report['forecast_2026_totals'] = {'selected': tot(pred)}
if ref_path.exists():
    report['forecast_2026_totals']['reference_sbg_gamma'] = tot(pd.read_csv(ref_path))
dec25 = sp[(sp.month == origin) & sp.paid].shape[0]
report['december_2025_observed'] = {
    'retained': dec25,
    'net_revenue': float(tx[tx.month == origin].net.sum()),
    'contribution': float(tx[tx.month == origin].contribution.sum())}
by_month = pred.groupby('calendar_month')[model.PREDICTIONS].sum()
report['forecast_by_month'] = {m: {k: float(v) for k, v in r.items()} for m, r in by_month.iterrows()}
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results' / 'final_fit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['origin_month', 'origin_survivors', 'origin_premium_share', 'forecast_2026_totals', 'december_2025_observed']}, indent=1))
print(json.dumps(report['cash'], indent=1))
for n, e in report['members'].items():
    print(n, e['converged'])
    for c in e['cells']:
        print(' ', c['channel'], c['primary_need'], c['cohort'][:7], round(c['alpha'], 3), round(c['beta'], 3), {k: round(v, 4) for k, v in c['monthly_churn_at_age'].items()})
