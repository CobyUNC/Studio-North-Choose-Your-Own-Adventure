"""Tier-aware member-composition model for Studio North's remaining 2026 value.

The model has two blocks, both estimated by penalized maximum likelihood on the
*person-level* renewal and ledger history (no MCMC, no fitted files):

1. Retention. A discrete-time logistic renewal hazard with cell covariates,
   a flexible tenure effect and partially pooled acquisition-cohort intercepts.
   Expected survivors in every channel × need × cohort cell are propagated from
   the observed origin survivors.

2. Cash per surviving member. Instead of one revenue curve per cell, monthly
   cash is built from the observed price book and the *current plan mix of the
   actual survivors*:
     - each survivor's plan tier at the origin is observed; standard members
       upgrade to premium with an estimated monthly hazard (premium never
       downgrades in the released history);
     - subscription receipts follow the posted tier prices;
     - add-on units per paid month follow a Poisson model by need, tier,
       tenure and calendar time, valued at the trending add-on price per unit;
     - refunds are a stable share of gross receipts;
     - subscription variable cost per tier follows its calendar trend; add-on
       variable cost is the observed cost per unit.

Contribution = net revenue − variable cost, before acquisition spending.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import panel


@dataclass(frozen=True)
class Config:
    # Retention hazard: 'sbg' (selected), 'bdw' or 'logit' (validation alternatives)
    hazard_kind: str = 'sbg'
    hazard_age_terms: tuple = ('log_age', 'first_renewal')
    hazard_interactions: bool = True       # need × channel, need × log tenure
    cohort_trend: bool = True               # linear acquisition-cohort trend
    cohort_ridge: float = 0.0               # precision of cohort intercepts (0 = none)
    cohort_rw: float = 0.0                  # random-walk precision across cohort effects (0 = none)
    heterogeneity: str = 'beta'             # 'beta' (sBG/BdW) or 'latent2' (two-class geometric)
    churn_event: str = 'all'                # 'all', or 'split' = cancelled and failed hazards fitted separately
    trial_halflife: float | None = None     # exponential recency weighting of renewal trials (months)
    period_effect: bool = False             # calendar drift fitted, frozen at the origin when forecasting
    period_forecast: str = 'frozen'         # 'frozen' (selected) or 'extrapolate' (sensitivity check)
    calendar_trend: bool = False            # linear calendar-month trend in hazard
    hazard_window: int | None = None        # use only trials in the last N months
    # Cash per member
    tier_aware: bool = True                 # survivor plan mix + upgrade hazard
    addon_calendar_trend: bool = True
    addon_price_trend: bool = True
    cost_trend: str = 'linear'              # 'linear', 'flat' or 'damped'
    cost_window: int = 6                    # months used for the cost trend
    damping: float = 0.9                    # monthly damping of cost trend if 'damped'
    cash_window: int = 12                   # months used for refund/add-on price ratios
    cash_anchor: float = 0.0                # prior member-months for a recent-level cash anchor (0 = off)
    anchor_months: int = 3
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Generalized linear models with a ridge penalty (deterministic, CPU, scipy)
# ----------------------------------------------------------------------------

def _fit_glm(X: np.ndarray, y: np.ndarray, n: np.ndarray, family: str,
             penalty: np.ndarray) -> np.ndarray:
    """Penalized MLE. Binomial: y successes of n trials. Poisson: y counts, n exposure."""
    X = np.asarray(X, float); y = np.asarray(y, float); n = np.asarray(n, float)
    offset = np.log(np.maximum(n, 1e-12)) if family == 'poisson' else 0.0

    def objective(b):
        eta = X @ b + offset
        if family == 'binomial':
            nll = (n * np.logaddexp(0.0, eta) - y * eta).sum()
            grad_eta = n / (1.0 + np.exp(-eta)) - y
        else:
            mu = np.exp(eta)
            nll = (mu - y * eta).sum()
            grad_eta = mu - y
        return nll + 0.5 * (penalty * b * b).sum(), X.T @ grad_eta + penalty * b

    start = np.zeros(X.shape[1])
    rate = y.sum() / max(n.sum(), 1e-12)
    start[0] = np.log(rate / (1 - rate)) if family == 'binomial' else np.log(max(rate, 1e-6))
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options={'maxiter': 5000, 'gtol': 1e-9, 'ftol': 1e-14})
    if not np.isfinite(result.x).all():
        raise RuntimeError(f'{family} GLM did not converge')
    return result.x


# ----------------------------------------------------------------------------
# Observed person-level history
# ----------------------------------------------------------------------------

def _history(tables: dict[str, pd.DataFrame], cutoff: str):
    origin = int(panel.month_index([cutoff])[0]) - 1
    acq = panel.acquired_customers(tables)
    acq = acq[acq.cohort_index <= origin].copy()
    sp = tables['subscription_periods']
    due = pd.to_datetime(sp.renewal_due_at.astype('string'), format='ISO8601', utc=True)
    sp = sp[(due < pd.Timestamp(cutoff, tz='UTC')) & sp.renewal_outcome.ne('pending')].copy()
    sp['month'] = panel.month_index(sp.period_start)
    sp = sp.merge(acq, on='person_id', validate='many_to_one')
    sp['age'] = sp.month - sp.cohort_index
    sp['paid'] = sp.renewal_outcome.eq('paid')
    sp['premium'] = sp.plan_tier.eq('premium').fillna(False).astype(bool)
    sp = sp.sort_values(['person_id', 'month']).reset_index(drop=True)
    tx = panel.ledger(tables)
    tx = tx[tx.month <= origin]
    return origin, acq, sp, tx


def _cell_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {'build': frame.primary_need.eq('build_skill').to_numpy(float),
            'partner': frame.channel.eq('partner').to_numpy(float)}


# ----------------------------------------------------------------------------
# Retention block
# ----------------------------------------------------------------------------

class Hazard:
    def __init__(self, config: Config, origin: int):
        self.c = config
        self.origin = origin

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        f = _cell_features(frame)
        age = frame.age.to_numpy(float)
        cols = [np.ones(len(frame)), f['build'], f['partner']]
        if self.c.hazard_interactions:
            cols.append(f['build'] * f['partner'])
        log_age = np.log(age)
        if 'log_age' in self.c.hazard_age_terms:
            cols.append(log_age)
            if self.c.hazard_interactions:
                cols.append(f['build'] * log_age)
        if 'first_renewal' in self.c.hazard_age_terms:
            cols.append((age == 1).astype(float))
        if 'inv_age' in self.c.hazard_age_terms:
            cols.append(1.0 / age)
        if self.c.cohort_trend:
            cols.append((frame.cohort_index.to_numpy(float) - 11.5) / 11.5)
        if self.c.calendar_trend:
            cols.append((frame.month.to_numpy(float) - 11.5) / 11.5)
        fixed = np.column_stack(cols)
        if self.c.cohort_ridge > 0:
            dummies = np.zeros((len(frame), self.origin + 1))
            dummies[np.arange(len(frame)), frame.cohort_index.to_numpy(int)] = 1.0
            fixed = np.column_stack([fixed, dummies])
        return fixed

    def fit(self, sp: pd.DataFrame) -> 'Hazard':
        trials = sp[sp.age >= 1]
        if self.c.hazard_window:
            trials = trials[trials.month > self.origin - self.c.hazard_window]
        grouped = (trials.groupby(['cohort_index', 'channel', 'primary_need', 'age', 'month'])
                   .agg(n=('paid', 'size'), paid=('paid', 'sum')).reset_index())
        X = self.design(grouped)
        n_fixed = X.shape[1] - (self.origin + 1 if self.c.cohort_ridge > 0 else 0)
        penalty = np.full(X.shape[1], 1e-4)
        penalty[0] = 0.0
        if self.c.cohort_ridge > 0:
            penalty[n_fixed:] = self.c.cohort_ridge
        churn = grouped.n - grouped.paid
        self.beta = _fit_glm(X, churn.to_numpy(), grouped.n.to_numpy(), 'binomial', penalty)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.design(frame) @ self.beta)))


class MixtureHazard:
    """Beta-mixed geometric (sBG, c = 1) or Beta-discrete-Weibull (BdW) renewal hazard.

    Each customer has a latent churn propensity θ ~ Beta(α, β) within a cell;
    under BdW the probability of surviving t renewals given θ is (1−θ)^(t^c)
    (Fader, Hardie, Liu, Davin & Steenburgh 2018). c < 1 adds individual-level
    negative duration dependence on top of the sorting that heterogeneity
    produces; c = 1 is the shifted-Beta-geometric. The Beta mean depends on
    need, channel, their interaction, a cohort trend and partially pooled
    cohort intercepts; the concentration on need and channel.
    """

    def __init__(self, config: Config, origin: int):
        self.c = config
        self.origin = origin
        self.weibull = config.hazard_kind == 'bdw'

    def _designs(self, frame):
        f = _cell_features(frame)
        cols = [np.ones(len(frame)), f['build'], f['partner']]
        if self.c.hazard_interactions:
            cols.append(f['build'] * f['partner'])
        if self.c.cohort_trend:
            cols.append((frame.cohort_index.to_numpy(float) - 11.5) / 11.5)
        xm = np.column_stack(cols)
        if self.c.cohort_ridge > 0 or self.c.cohort_rw > 0:
            d = np.zeros((len(frame), self.origin + 1))
            d[np.arange(len(frame)), frame.cohort_index.to_numpy(int)] = 1.0
            xm = np.column_stack([xm, d])
        xc = np.column_stack([np.ones(len(frame)), f['build'], f['partner']])
        xw = np.column_stack([np.ones(len(frame)), f['build']])
        return xm, xc, xw

    def _hazard(self, params, xm, xc, xw, age, period=None):
        from scipy.special import betaln, expit
        km, kc = xm.shape[1], xc.shape[1]
        if self.c.period_effect:
            gamma, params = params[-1], params[:-1]
        if self.c.heterogeneity == 'latent2':
            eta = xm @ params[:km]
            pi = expit(xc @ params[km:km + kc])
            th1, th2 = expit(eta), expit(eta - np.exp(params[km + kc]))
            s1, s2 = pi * (1 - th1) ** (age - 1), (1 - pi) * (1 - th2) ** (age - 1)
            h = (s1 * th1 + s2 * th2) / (s1 + s2)
        else:
            h = self._beta_hazard(params, xm, xc, xw, age, km, kc)
        if self.c.period_effect and period is not None:
            h = -np.expm1(np.exp(gamma * period) * np.log1p(-np.clip(h, 1e-12, 1 - 1e-12)))
        return np.clip(h, 1e-12, 1 - 1e-12)

    def _beta_hazard(self, params, xm, xc, xw, age, km, kc):
        from scipy.special import betaln, expit
        mu = expit(xm @ params[:km])
        phi = np.exp(xc @ params[km:km + kc])
        a, b = mu * phi, (1 - mu) * phi
        if self.weibull:
            cw = np.exp(xw @ params[km + kc:])
            t1, t0 = age ** cw, (age - 1) ** cw
        else:
            t1, t0 = age, age - 1
        return -np.expm1(betaln(a, b + t1) - betaln(a, b + t0))

    def fit(self, sp: pd.DataFrame, event: str = 'all') -> 'MixtureHazard':
        trials = sp[sp.age >= 1].copy()
        trials['event'] = (~trials.paid if event == 'all' else trials.renewal_outcome.eq(event)).astype(float)
        trials['w'] = (0.5 ** ((self.origin - trials.month) / self.c.trial_halflife)
                       if self.c.trial_halflife else 1.0)
        trials['wn'] = trials.w
        trials['we'] = trials.w * trials.event
        g = (trials.groupby(['cohort_index', 'channel', 'primary_need', 'age'])
             .agg(n=('wn', 'sum'), churn=('we', 'sum')).reset_index())
        xm, xc, xw = self._designs(g)
        age = g.age.to_numpy(float); n = g.n.to_numpy(float); churn = g.churn.to_numpy(float)
        period = (g.cohort_index.to_numpy(float) + age - self.origin) / 12.0
        km, kc = xm.shape[1], xc.shape[1]
        kw = (xw.shape[1] if self.weibull else 0) + (1 if self.c.heterogeneity == 'latent2' else 0)
        kw += 1 if self.c.period_effect else 0
        k = self.origin + 1
        n_fixed = km - (k if (self.c.cohort_ridge > 0 or self.c.cohort_rw > 0) else 0)
        P = np.zeros((km + kc + kw, km + kc + kw))
        idx = np.arange(km + kc + kw)
        P[idx, idx] = 1e-4
        P[0, 0] = 0.0
        if n_fixed < km:
            block = np.diag(np.full(k, self.c.cohort_ridge + 1e-4))
            if self.c.cohort_rw > 0:
                D = np.diff(np.eye(k), axis=0)
                block = block + self.c.cohort_rw * D.T @ D
            P[n_fixed:km, n_fixed:km] = block

        def nll(p):
            h = self._hazard(p, xm, xc, xw, age, period)
            return -(churn * np.log(h) + (n - churn) * np.log1p(-h)).sum() + 0.5 * p @ P @ p

        start = np.zeros(km + kc + kw)
        start[0] = -2.0
        start[km] = 0.0 if self.c.heterogeneity == 'latent2' else np.log(4.0)
        result = minimize(nll, start, method='L-BFGS-B',
                          options={'maxiter': 20000, 'maxfun': 200000, 'ftol': 1e-13, 'gtol': 1e-8})
        if not np.isfinite(result.x).all():
            raise RuntimeError('Retention mixture did not converge')
        self.params = result.x
        self.converged = bool(result.success)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        xm, xc, xw = self._designs(frame)
        # Calendar drift is held at its origin level (period = 0) unless the
        # sensitivity option asks to extend it.
        period = None
        if self.c.period_effect and self.c.period_forecast == 'extrapolate':
            period = (frame.month.to_numpy(float) - self.origin) / 12.0
        return self._hazard(self.params, xm, xc, xw, frame.age.to_numpy(float), period)


class SplitHazard:
    """Voluntary (cancelled) and involuntary (failed payment) churn as competing risks."""

    def __init__(self, config: Config, origin: int):
        self.parts = [MixtureHazard(config, origin), MixtureHazard(config, origin)]

    def fit(self, sp):
        self.parts[0].fit(sp, 'cancelled'); self.parts[1].fit(sp, 'failed')
        self.converged = all(p.converged for p in self.parts)
        return self

    def predict(self, frame):
        return np.clip(self.parts[0].predict(frame) + self.parts[1].predict(frame), 0, 1 - 1e-9)


# ----------------------------------------------------------------------------
# Cash block
# ----------------------------------------------------------------------------

def _trend(months: np.ndarray, values: np.ndarray, weights: np.ndarray, target: np.ndarray,
           kind: str, damping: float, origin: int) -> np.ndarray:
    """Weighted linear trend through the recent window, extrapolated to target months."""
    if kind == 'flat' or len(np.unique(months)) < 3:
        level = np.average(values[months == months.max()], weights=weights[months == months.max()])
        return np.full(len(target), level)
    A = np.column_stack([np.ones(len(months)), months - origin])
    w = np.sqrt(weights)
    coef = np.linalg.lstsq(A * w[:, None], values * w, rcond=None)[0]
    steps = target - origin
    if kind == 'damped':
        steps = np.array([sum(damping ** k for k in range(1, s + 1)) for s in steps])
    return coef[0] + coef[1] * steps


class Cash:
    def __init__(self, config: Config, origin: int):
        self.c = config
        self.origin = origin

    def fit(self, sp: pd.DataFrame, tx: pd.DataFrame) -> 'Cash':
        c, origin = self.c, self.origin
        paid = sp[sp.paid & (sp.age >= 1)].copy()
        pay = tx[tx.transaction_type.eq('payment')]
        subs = pay[pay.charge_type.eq('subscription')]
        adds = pay[pay.charge_type.eq('add_on')]
        # Posted subscription prices by tier (renewal months carry no discount).
        tier = sp.set_index('period_id').premium
        s = subs.assign(premium=subs.period_id.map(tier))
        s = s[s.period_id.isin(paid.period_id)]
        self.price = {t: float(g.gross_amount.median()) for t, g in s.groupby('premium')}
        self.price.setdefault(True, 75.0); self.price.setdefault(False, 50.0)
        # Subscription variable cost per tier: calendar trend over a recent window.
        recent = s[s.month > origin - c.cost_window]
        self.cost = {}
        for t in (False, True):
            g = recent[recent.premium == t].groupby('month').variable_cost.agg(['mean', 'size'])
            self.cost[t] = (g.index.to_numpy(float), g['mean'].to_numpy(float), g['size'].to_numpy(float))
        # Refund share of gross receipts (renewal months).
        window = tx[(tx.month > origin - c.cash_window)]
        window = window[window.period_id.isin(paid.period_id)]
        gross = float((window.gross_amount - window.discount_amount).sum())
        self.refund_share = float(window.refund_amount.sum()) / gross if gross > 0 else 0.0
        # Add-on units per paid renewal month: Poisson GLM.
        units = adds.groupby('period_id').quantity.sum()
        paid['units'] = paid.period_id.map(units).fillna(0).astype(float)
        grouped = (paid.groupby(['channel', 'primary_need', 'premium', 'age', 'month'])
                   .agg(n=('units', 'size'), units=('units', 'sum')).reset_index())
        X = self._addon_design(grouped)
        penalty = np.r_[0.0, np.full(X.shape[1] - 1, 1e-4)]
        self.addon_beta = _fit_glm(X, grouped.units.to_numpy(), grouped.n.to_numpy(), 'poisson', penalty)
        # Add-on price and variable cost per unit.
        a = adds[adds.month > origin - c.cash_window]
        g = a.groupby('month').agg(gross=('gross_amount', 'sum'), units=('quantity', 'sum'),
                                   cost=('variable_cost', 'sum'))
        self.addon_price = (g.index.to_numpy(float), (g.gross / g.units).to_numpy(float), g.units.to_numpy(float))
        self.addon_cost = float(g.cost.sum() / g.units.sum())
        # Upgrade hazard (standard -> premium) among renewing standard members.
        prev = sp[['person_id', 'month', 'premium']].assign(month=sp.month + 1).rename(columns={'premium': 'prev_premium'})
        renewals = paid.merge(prev, on=['person_id', 'month'], how='inner')
        std = renewals[~renewals.prev_premium.astype(bool)]
        grouped = (std.groupby(['channel', 'primary_need', 'age', 'month'])
                   .agg(n=('premium', 'size'), up=('premium', 'sum')).reset_index())
        X = self._upgrade_design(grouped)
        penalty = np.r_[0.0, np.full(X.shape[1] - 1, 1e-4)]
        self.upgrade_beta = _fit_glm(X, grouped.up.to_numpy(float), grouped.n.to_numpy(float), 'binomial', penalty)
        if (renewals.prev_premium.astype(bool) & ~renewals.premium).any():
            self.downgrade_rate = float((renewals.prev_premium.astype(bool) & ~renewals.premium).sum()
                                        / max(renewals.prev_premium.astype(bool).sum(), 1))
        else:
            self.downgrade_rate = 0.0
        return self

    def _addon_design(self, frame):
        f = _cell_features(frame)
        cols = [np.ones(len(frame)), f['build'], f['partner'], frame.premium.to_numpy(float),
                np.log(frame.age.to_numpy(float))]
        if self.c.addon_calendar_trend:
            cols.append((frame.month.to_numpy(float) - self.origin) / 12)
        return np.column_stack(cols)

    def _upgrade_design(self, frame):
        f = _cell_features(frame)
        return np.column_stack([np.ones(len(frame)), f['build'], f['partner'],
                                np.log(frame.age.to_numpy(float))])

    def addon_units(self, frame):
        return np.exp(self._addon_design(frame) @ self.addon_beta)

    def upgrade(self, frame):
        return 1.0 / (1.0 + np.exp(-(self._upgrade_design(frame) @ self.upgrade_beta)))

    def sub_cost(self, premium: bool, months: np.ndarray) -> np.ndarray:
        m, v, w = self.cost[premium]
        return _trend(m, v, w, months, self.c.cost_trend, self.c.damping, self.origin)

    def addon_unit_price(self, months: np.ndarray) -> np.ndarray:
        m, v, w = self.addon_price
        return _trend(m, v, w, months, 'linear' if self.c.addon_price_trend else 'flat',
                      self.c.damping, self.origin)


# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------

# Final specification (second pass), fixed before the submission run. Four
# sBG retention members are averaged with equal weights, sharing one
# tier-aware cash block. They span two unresolved identification choices:
#   * cohort quality: a linear acquisition-cohort trend, or none;
#   * calendar drift: none, or a calendar-time multiplier on the hazard that is
#     estimated jointly and then held at its December-2025 level (never
#     extrapolated). With two years of data, tenure, cohort and calendar
#     effects are not separately identified, so the ensemble hedges across them.
SELECTED = {'sbg_cohort_trend': Config(hazard_kind='sbg', cohort_trend=True),
            'sbg_no_cohort_effect': Config(hazard_kind='sbg', cohort_trend=False),
            'sbg_cohort_trend_period': Config(hazard_kind='sbg', cohort_trend=True, period_effect=True),
            'sbg_no_cohort_effect_period': Config(hazard_kind='sbg', cohort_trend=False, period_effect=True)}
FIRST_PASS = {k: SELECTED[k] for k in ['sbg_cohort_trend', 'sbg_no_cohort_effect']}
PREDICTIONS = ['predicted_retained', 'predicted_net_revenue', 'predicted_contribution']


def forecast_ensemble(tables: dict[str, pd.DataFrame], grid: pd.DataFrame,
                      members: dict[str, Config] = SELECTED) -> pd.DataFrame:
    """Equal-weight average of expected values (preserves monotone membership)."""
    runs = [forecast(tables, grid, cfg) for cfg in members.values()]
    out = runs[0].copy()
    out[PREDICTIONS] = np.mean([r[PREDICTIONS].to_numpy(float) for r in runs], axis=0)
    return out


def _member_cash(cash: 'Cash', frame: pd.DataFrame, tier: bool) -> tuple[np.ndarray, np.ndarray]:
    """Expected net revenue and contribution per paid member of one tier."""
    months = frame.month.to_numpy()
    units = cash.addon_units(frame.assign(premium=float(tier)))
    gross = cash.price[tier] + units * cash.addon_unit_price(months)
    member_net = gross * (1 - cash.refund_share)
    return member_net, member_net - cash.sub_cost(tier, months) - units * cash.addon_cost


def _cash_anchor(cash, sp, tx, key, origin, config) -> tuple[float, float]:
    """Shrunk ratio of a cell's actual to modelled cash over its last renewal months.

    Expected cash uses the cell's *actual* paid members by tier in each month, so the
    ratio isolates cell-specific add-on/refund behaviour. Acquisition months (age 0)
    are excluded because they carry first-month discounts.
    """
    cell = sp[(sp.cohort_month == key[0]) & (sp.channel == key[1]) & (sp.primary_need == key[2]) & sp.paid
              & (sp.month > origin - config.anchor_months) & (sp.age >= 1)]
    if cell.empty:
        return 1.0, 1.0
    counts = cell.groupby(['month', 'age', 'premium']).size().reset_index(name='n')
    exp_net = exp_con = 0.0
    for r in counts.itertuples(index=False):
        f = pd.DataFrame({'channel': [key[1]], 'primary_need': [key[2]], 'age': [r.age], 'month': [r.month]})
        mn, mc = _member_cash(cash, f, bool(r.premium))
        exp_net += r.n * float(mn[0]); exp_con += r.n * float(mc[0])
    people = set(cell.person_id)
    t = tx[tx.person_id.isin(people) & tx.month.isin(set(cell.month))]
    t = t.merge(cell[['person_id', 'month']], on=['person_id', 'month'])
    w = float(len(cell)); k = w / (w + config.cash_anchor)
    act_net, act_con = float(t.net.sum()), float(t.contribution.sum())
    f_net = 1 + k * (act_net / exp_net - 1) if exp_net > 0 else 1.0
    f_con = 1 + k * (act_con / exp_con - 1) if exp_con > 0 else 1.0
    return max(f_net, 0.0), max(f_con, 0.0)


def forecast(tables: dict[str, pd.DataFrame], grid: pd.DataFrame, config: Config = Config()) -> pd.DataFrame:
    grid = grid.copy()
    first_future = int(panel.month_index(grid.calendar_month).min())
    cutoff = panel.index_to_month([first_future])[0]
    origin, acq, sp, tx = _history(tables, cutoff)
    if config.hazard_kind == 'logit':
        hazard = Hazard(config, origin).fit(sp)
    elif config.churn_event == 'split':
        hazard = SplitHazard(config, origin).fit(sp)
    else:
        hazard = MixtureHazard(config, origin).fit(sp)
    cash = Cash(config, origin).fit(sp, tx)

    # Survivors at the origin and their plan tier.
    survivors = sp[(sp.month == origin) & sp.paid]
    mix = (survivors.groupby(panel.CELL).agg(n=('person_id', 'size'), premium=('premium', 'sum'))
           .reset_index())
    out = []
    for key, g in grid.groupby(panel.CELL, sort=True):
        g = g.sort_values('age_month').copy()
        n0 = float(g.origin_retained.iloc[0])
        row = mix[(mix.cohort_month == key[0]) & (mix.channel == key[1]) & (mix.primary_need == key[2])]
        observed_n = float(row.n.iloc[0]) if len(row) else 0.0
        if abs(observed_n - n0) > 1e-9:
            raise ValueError(f'Origin survivors disagree with the grid for {key}: {observed_n} vs {n0}')
        p_premium = float(row.premium.iloc[0]) / observed_n if observed_n > 0 else 0.0
        if not config.tier_aware:
            # Cell-level cash: use the cell's historical renewal plan mix, not the survivors'.
            hist = sp[(sp.cohort_month == key[0]) & (sp.channel == key[1]) & (sp.primary_need == key[2]) & sp.paid]
            p_premium = float(hist.premium.mean()) if len(hist) else 0.0
        months = panel.month_index(g.calendar_month)
        frame = pd.DataFrame({'channel': key[1], 'primary_need': key[2],
                              'cohort_index': int(panel.month_index([key[0]])[0]),
                              'age': g.age_month.to_numpy(int), 'month': months})
        h = hazard.predict(frame)
        retained = n0 * np.cumprod(1.0 - h)
        up = cash.upgrade(frame) if config.tier_aware else np.zeros(len(frame))
        premium_share = np.empty(len(frame))
        p = p_premium
        for j in range(len(frame)):
            p = p + (1 - p) * up[j] - p * cash.downgrade_rate * config.tier_aware
            premium_share[j] = p
        net = np.zeros(len(frame)); contribution = np.zeros(len(frame))
        for tier, share in ((True, premium_share), (False, 1 - premium_share)):
            member_net, member_contribution = _member_cash(cash, frame, tier)
            net += share * member_net
            contribution += share * member_contribution
        if config.cash_anchor > 0:
            f_net, f_con = _cash_anchor(cash, sp, tx, key, origin, config)
            net, contribution = net * f_net, contribution * f_con
        g['predicted_retained'] = retained
        g['predicted_net_revenue'] = retained * net
        g['predicted_contribution'] = retained * contribution
        out.append(g)
    result = pd.concat(out).sort_values(panel.KEYS).reset_index(drop=True)
    # No clipping: a fit that violates the business rules is an error, not a forecast.
    if (result.predicted_contribution < 0).any():
        raise RuntimeError('Negative expected member contribution; inspect the cash fit')
    if (result.predicted_contribution > result.predicted_net_revenue + 1e-9).any():
        raise RuntimeError('Expected contribution exceeds net revenue; inspect the cost fit')
    return result
