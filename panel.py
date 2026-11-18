"""Observed-data preparation shared by the forecast and its historical validation.

The functions reproduce the released lab's canonical SQL definitions
(`sql/01_views.sql`) in pandas so that a team submission needs no database:

* a customer's acquisition cohort is the month of the first payment with
  positive gross-minus-discount amount;
* channel comes from the credited campaign (falling back to source_type);
* a customer is *retained* in a calendar month when a subscription period that
  starts in that month has a ``paid`` outcome and a renewal date before the
  data cutoff;
* net revenue and contribution are the sums of all ledger rows (payments and
  refunds) that occur in the calendar month.

``truncate_tables`` recreates what would have been released at an earlier
cutoff. It is used only by the chronological validation; the submitted
``predict`` receives the instructor's tables unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CELL = ['cohort_month', 'channel', 'primary_need']
KEYS = CELL + ['calendar_month', 'age_month']


def month_index(values) -> np.ndarray:
    """Months since January 2024 for ISO date/timestamp strings."""
    ts = pd.to_datetime(pd.Series(values).astype('string').str.slice(0, 10))
    return ((ts.dt.year - 2024) * 12 + ts.dt.month - 1).to_numpy(int)


def index_to_month(index) -> pd.Series:
    index = np.asarray(index, int)
    years, months = 2024 + index // 12, index % 12 + 1
    return pd.Series([f'{y:04d}-{m:02d}-01' for y, m in zip(years, months)])


def truncate_tables(tables: dict[str, pd.DataFrame], cutoff: str) -> dict[str, pd.DataFrame]:
    """Records that a release with an exclusive cutoff would have contained.

    Renewal outcomes due on or after the cutoff are unknown, so those periods
    are dropped (the model never uses pending rows). Ledger, events and
    feedback are cut at their own timestamps; acquisition credit at
    ``credited_at``. Campaign spending is cut at ``spend_date`` and holdout
    tests at ``followup_end_at``. People are kept only if acquired before the
    cutoff.
    """
    c = pd.Timestamp(cutoff, tz='UTC')
    ts = lambda s: pd.to_datetime(s.astype('string'), format='ISO8601', utc=True)
    out = {}
    sp = tables['subscription_periods']
    out['subscription_periods'] = sp[ts(sp.renewal_due_at) < c].copy()
    tx = tables['transactions']
    out['transactions'] = tx[ts(tx.occurred_at) < c].copy()
    pe = tables['product_events']
    out['product_events'] = pe[ts(pe.occurred_at) < c].copy()
    fb = tables['feedback']
    out['feedback'] = fb[ts(fb.submitted_at) < c].copy()
    cs = tables['campaign_spend']
    out['campaign_spend'] = cs[ts(cs.spend_date) < c].copy()
    ch = tables['channel_holdouts']
    out['channel_holdouts'] = ch[ts(ch.followup_end_at) < c].copy()
    aa = tables['acquisition_attribution']
    out['acquisition_attribution'] = aa[ts(aa.credited_at) < c].copy()
    keep = set(out['acquisition_attribution'].person_id)
    out['people'] = tables['people'][tables['people'].person_id.isin(keep)].copy()
    return out


def acquired_customers(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tx = tables['transactions']
    pay = tx[(tx.transaction_type == 'payment') & ((tx.gross_amount - tx.discount_amount) > 0)]
    pay = pay.sort_values(['person_id', 'occurred_at', 'transaction_id'])
    first = pay.drop_duplicates('person_id')[['person_id', 'occurred_at']]
    campaigns = (tables['campaign_spend'].groupby('campaign_id')
                 .agg(channel=('channel', 'min')).reset_index())
    a = (first.merge(tables['people'], on='person_id', validate='one_to_one')
         .merge(tables['acquisition_attribution'], on='person_id', validate='one_to_one')
         .merge(campaigns, left_on='credited_campaign_id', right_on='campaign_id', how='left'))
    a['channel'] = a.channel.fillna(a.source_type).astype(str)
    a['cohort_index'] = month_index(a.occurred_at)
    a['cohort_month'] = index_to_month(a.cohort_index).to_numpy()
    cols = ['person_id', 'cohort_month', 'cohort_index', 'channel', 'primary_need',
            'age_band', 'household_type', 'region']
    return a[cols].astype({'primary_need': str, 'age_band': str, 'household_type': str,
                           'region': str}).sort_values('person_id').reset_index(drop=True)


def ledger(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transactions with month index and the canonical cash measures."""
    tx = tables['transactions'].copy()
    tx['month'] = month_index(tx.occurred_at)
    pay = tx.transaction_type.eq('payment')
    tx['gross_payment'] = np.where(pay, tx.gross_amount - tx.discount_amount, 0.0)
    tx['net'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    tx['contribution'] = tx.net - tx.variable_cost
    return tx


def paid_periods(tables: dict[str, pd.DataFrame], cutoff: str) -> pd.DataFrame:
    sp = tables['subscription_periods']
    due = pd.to_datetime(sp.renewal_due_at.astype('string'), format='ISO8601', utc=True)
    sp = sp[(due < pd.Timestamp(cutoff, tz='UTC')) & sp.renewal_outcome.eq('paid')].copy()
    sp['month'] = month_index(sp.period_start)
    return sp


def cohort_age_summary(tables: dict[str, pd.DataFrame], cutoff: str) -> pd.DataFrame:
    """Cell × calendar-month history, identical in meaning to the lab view."""
    origin = month_index([cutoff])[0] - 1
    acq = acquired_customers(tables)
    acq = acq[acq.cohort_index <= origin]
    pp = paid_periods(tables, cutoff)
    retained = (pp[['person_id', 'month']].drop_duplicates().assign(retained=1))
    tx = ledger(tables)
    cash = tx.groupby(['person_id', 'month'])[['gross_payment', 'net', 'contribution']].sum().reset_index()
    # Person × month grid from acquisition month to the origin month.
    span = origin - acq.cohort_index.to_numpy() + 1
    pm = pd.DataFrame({'person_id': np.repeat(acq.person_id.to_numpy(), span)})
    pm['month'] = np.concatenate([np.arange(c, origin + 1) for c in acq.cohort_index])
    pm = (pm.merge(acq[['person_id'] + CELL + ['cohort_index']], on='person_id')
            .merge(retained, on=['person_id', 'month'], how='left')
            .merge(cash, on=['person_id', 'month'], how='left'))
    pm = pm.fillna({'retained': 0, 'gross_payment': 0.0, 'net': 0.0, 'contribution': 0.0})
    out = (pm.groupby(CELL + ['cohort_index', 'month'])
             .agg(n_acquired=('person_id', 'size'), n_retained=('retained', 'sum'),
                  gross_payment_revenue=('gross_payment', 'sum'), net_revenue=('net', 'sum'),
                  contribution=('contribution', 'sum')).reset_index())
    out['age_month'] = out.month - out.cohort_index
    out['calendar_month'] = index_to_month(out.month).to_numpy()
    out['n_retained'] = out.n_retained.astype(int)
    return out.sort_values(CELL + ['age_month']).reset_index(drop=True)


def prediction_grid(summary: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Future keys from the last observed month of every cell (lab definition)."""
    origin = summary.month.max()
    last = summary[summary.month == origin]
    rows = []
    for r in last.itertuples(index=False):
        for step in range(1, horizon + 1):
            rows.append({'cohort_month': r.cohort_month, 'channel': r.channel,
                         'primary_need': r.primary_need,
                         'calendar_month': index_to_month([origin + step])[0],
                         'age_month': int(r.age_month) + step, 'n_acquired': int(r.n_acquired),
                         'origin_retained': int(r.n_retained)})
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
