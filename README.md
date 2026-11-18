# Studio North: Choose Your Own Adventure: a tier-aware member-composition forecast

Team submission for the optional Studio North private-holdout model challenge
(`submission-contract.json`, contract `studio-north-forecast/1.0`, dataset 5.1).

**In one sentence:** we keep the shifted-Beta-geometric (sBG) idea for *who
stays*, but replace the reference's cell-level Gamma revenue curve with a
structural model of *what the actual survivors pay*. That model follows each
survivor's observed plan tier, upgrades, add-ons, refunds and the falling
variable cost. In chronological validation it cut cohort-month contribution
WAPE from **9.2% (reference) to 6.3%**, and it beat the reference at all six
historical origins.

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

The command fits the model from the eight observed CSVs and writes the 1,152
forecast rows plus a `.receipt.json` (input, code, lock and output hashes).
On the 4-core development machine the full fit and forecast took about
**3 seconds**, against the 60-minute contract budget. It needs no GPU, no
network and no precomputed artifacts. The model is deterministic: the seed is
accepted and recorded, and repeated runs produce identical bytes (see
[Reproducibility](#reproducibility-evidence)).

---

## 1. What is being forecast

For each of the 24 existing acquisition cohorts × channel × stated need (96
cells), we forecast **expected retained members, net revenue and
contribution in every calendar month of January–December 2026**.

This is *remaining value over a common calendar window*, not the report's
"first twelve receipts" LTV:

| | Report LTV | This challenge |
|---|---|---|
| Window | Each cohort's first 12 monthly receipts | Jan–Dec 2026 for every cohort |
| Ages covered | 0–11 for every cohort | Jan 2024 cohort: ages 24–35. Dec 2025 cohort: ages 1–12 |
| Acquisition cost | Netted against value (payback) | Not subtracted; CAC predates the window |
| Main risk | Young cohorts' early churn | Old cohorts' extrapolation beyond the oldest observed age (23) |

Every cohort has exactly twelve future months. Younger cohorts have a
larger *forecast share* of their trajectory, not more forecast months.
Contribution is net revenue minus refunds and nonmarketing variable costs,
before acquisition spending. The score is cohort-month contribution WAPE
after summing the four channel/need cells, over 288 points.

## 2. LTV model landscape (research summary)

Studio North is a **contractual** business. Churn is observed (a renewal is
paid, cancelled or failed), so models built for non-contractual settings
(Pareto/NBD, BG/NBD, Gamma-Gamma spend) do not apply directly. The relevant
families are:

| Family | Idea | Strengths | Weaknesses here |
|---|---|---|---|
| **sBG** (Fader & Hardie 2007) | Each customer has a constant churn probability θ ~ Beta(α, β). The aggregate hazard falls as high-θ customers leave | Two parameters per segment; principled extrapolation; well documented on subscription data | Ignores individual duration dependence; says nothing about revenue per member |
| **Beta-discrete-Weibull** (Fader, Hardie, Liu, Davin & Steenburgh 2018) | sBG plus individual duration dependence c (θ applied to t^c) | Separates sorting from "lock-in" | One more parameter; weakly identified with short histories |
| **Discrete-time hazard regression** (logit or cloglog on person-periods) | Flexible tenure terms plus covariates, including time-varying ones | Uses customer-level signals; easy to fit | Parametric tenure terms extrapolate poorly; time-varying covariates must themselves be forecast |
| **Hierarchical Bayesian sBG + Gamma revenue** (released reference, adapting Orduz 2024) | Cohort-pooled sBG retention; Gamma revenue per retained member with a log-age mean | Coherent uncertainty; pools sparse cells | Revenue per member is a smooth age curve at cell level, blind to *which* plans the survivors hold |
| **Machine learning / neural** (gradient boosting, Bayesian neural nets as in Orduz) | Learn hazard or revenue surfaces | Captures interactions | Flat or erratic outside observed ages; needs much more data than 5,703 customers |
| **Customer-base valuation** (Gupta, Lehmann & Stuart 2004; Fader & Hardie 2010; McCarthy, Fader & Hardie 2017) | Retention × margin, summed over the remaining base | Links CLV to firm value | Margin is usually assumed constant per customer |

The literature's main warning, which the reference already respects, is that
heterogeneity makes the aggregate hazard fall with tenure. Ignoring it
under-values the remaining base (Fader & Hardie 2010). Our exploratory work
showed that **for this business the larger forecasting error sits on the
margin side**. That is where we put the alternative.

## 3. The reference model and why we changed the cash block

**Reference assumptions.** These come from the released `cohort_lab/model.py`.
Retention is an sBG renewal hazard μφ/(φ + age − 1). The mean μ depends on
need, channel, a cohort trend and partially pooled cohort effects; the
concentration φ depends on need and channel. Gross revenue is Gamma with
mean per retained member exp(β·[log1p(age), cohort trend, need, channel]).
Net and contribution come from age-aligned historical net/gross and
contribution/gross ratios, falling back to the all-age average beyond the
oldest observed age. Fitting uses NUTS on CPU.

**What the released records show** (development data only, all before 2026-01-01):

1. **Plan tier drives margin.** Renewals cost $50 (standard) or $75
   (premium). Variable cost is about $29 or $39 per month, so contribution
   per member is roughly $25 versus $40.
2. **Standard members upgrade, and premium never downgrades.** About 7–8%
   of renewing standard members move to premium each month, at every tenure.
   The survivors' premium share rose from 20% at acquisition to 55% in
   December 2025.
3. **Tier does not predict churn** once tenure, need, channel and cohort are
   controlled for (logit coefficient −0.03, z = −0.7). Product events,
   support contacts, feedback ratings and demographics were also not
   predictive (all |z| < 1.6). Retention depends on need, channel, cohort and
   tenure.
4. **Variable cost per subscription falls steadily with calendar time.**
   Standard fell from $31.00 to $28.87 and premium from $40.50 to $38.76 over
   2024–25. Add-on units per paid month rise with tenure and calendar time,
   and add-on price per unit drifts upward ($9.47 to $10.89) as the $12
   add-on gains share. Refunds are a stable ~1.1% of gross receipts (25% of
   a payment, ~6% of periods, same month).

A cell-level revenue curve sees the *average* of a shifting tier mix and of
falling costs. Its historical contribution/gross ratios also embed costs
that are higher than the costs that will apply in 2026. The backtest
decomposition below confirms the result: the reference forecasts retention
well but under-forecasts **contribution per retained member by 6–9% at
every origin**.

### Alternatives considered

We stated three candidates, with assumptions, before fitting:

- **A. Person-level logistic discrete-time hazard.** Uses log-tenure, a
  first-renewal indicator, need × channel, need × tenure, a cohort trend and
  ridge-pooled cohort intercepts. Behavioural covariates were dropped
  because they had no predictive power (finding 3).
- **B. Beta-discrete-Weibull retention.** Tests whether individual-level
  duration dependence improves extrapolation.
- **C. sBG retention (as the reference) + tier-aware cash block.** Selected.

Each retention form was paired with the same cash block, so the comparison
isolates retention. Ablations then remove one cash component at a time.

## 4. Selected model

All parameters are estimated at run time from the supplied tables by
penalized maximum likelihood (scipy L-BFGS). Code is in `model.py`; the data
preparation in `panel.py` reproduces the lab's canonical SQL and matched the
lab's `cohort_age_summary` on all 1,200 cell-months.

### Retention: sBG ensemble

For a cell with need n, channel c and cohort k, renewal t (t ≥ 1) fails with
probability

  h(t) = 1 − B(α, β + t) / B(α, β + t − 1) = α / (α + β + t − 1),

where α = μφ and β = (1 − μ)φ. Two specifications tied in validation. Their
forecasts are **averaged with equal weights**, which keeps expected
membership monotone:

| Member | logit μ | log φ |
|---|---|---|
| `sbg_cohort_trend` | need + channel + need×channel + linear cohort trend | need + channel |
| `sbg_no_cohort_effect` | need + channel + need×channel | need + channel |

The two members disagree about whether recent cohorts' better early
retention will persist, so the average hedges a real identification problem.
In validation the trend member won at four origins and the no-cohort member
at two, including the 2025-01 origin where the trend member missed most. Expected survivors are propagated from each cell's *observed*
December 2025 survivors (`origin_retained`): S(m) = S(m−1)·(1 − h(age_m)).

### Cash per survivor: tier-aware composition

For each cell we read the **plan tier of every December 2025 survivor**
(premium share p₀). Because tier does not affect churn, the survivors' tier
mix evolves only through upgrades:

  p(m) = p(m−1) + (1 − p(m−1))·u(age_m),

where u is a logistic upgrade hazard on need, channel and log-tenure (about
7% per month). Expected monthly receipts per survivor of tier τ are:

- net(τ, m) = [price_τ + λ(τ, need, channel, age, m) · addon_price(m)] · (1 − refund_share)
- contribution(τ, m) = net(τ, m) − cost_τ(m) − λ · $3.50 per add-on unit

Where:
- price_τ: the posted renewal prices ($50 and $75). Renewal months carry no discounts.
- λ: add-on units per paid month, from a Poisson regression on need, channel,
  tier, log-tenure and calendar month.
- addon_price(m): the linear calendar trend of add-on revenue per unit over
  the last 12 months.
- refund_share: refunds ÷ gross over the last 12 months (1.1%).
- cost_τ(m): the linear calendar trend of subscription variable cost per
  tier over the last 6 months.

The cell's forecasts are S(m)·Σ_τ share_τ(m)·net(τ, m) and the matching
contribution. Expected cash is zero whenever expected membership is zero.
The code raises an error rather than clipping if a forecast would break a
business rule.

### Final fitted settings (December 2025 origin)

Full detail is in `validation/results/final_fit.json`.

| Quantity | Value |
|---|---|
| Survivors at origin / premium share | 3,007 / 55.2% |
| Upgrade hazard (age 6 → 24) | 7.2% → 6.6% per month |
| Standard variable cost: Dec 2025 → Jan 2026 → Dec 2026 | $28.87 → $28.80 → $28.16 |
| Premium variable cost: Dec 2025 → Jan 2026 → Dec 2026 | $38.76 → $38.68 → $37.96 |
| Add-on price per unit, Jan → Dec 2026 | $10.84 → $11.33 |
| Refund share of gross | 1.11% |
| sBG search/save-time, Jan 2024 cohort (trend member): α, β | 0.84, 2.19 |
| Monthly churn at age 24 → 35, same cell (trend / no-cohort member) | 3.2% → 2.3% / 2.7% → 1.9% |
| Monthly churn at age 24 → 35, partner/build-skill (trend / no-cohort member) | 2.0% → 1.5% / 1.8% → 1.3% |

**2026 totals.** We forecast **$1.045M contribution** on $2.115M net
revenue and 27,937 member-months. The reference forecasts $1.008M on
$2.147M and 28,806 member-months. Our forecast has slightly fewer members
but more contribution per member, which matches the reference's backtest
bias.

### Extrapolation beyond observed ages (24–35)

Only the January 2024 cohort reaches age 23 in the data. Its 2026 forecasts
are for ages 24–35. The assumptions behind them:

- **Retention.** sBG keeps the hazard falling as α/(α + β + t − 1). That
  decline comes only from heterogeneity sorting, so it slows smoothly and
  never turns negative or explodes. The observed hazards at ages 16–23
  (1–4%) fit this shape, and the model implies 1.3–3.2% at ages 24–35. We
  rejected the logit tenure curve partly because a log-age term with no
  heterogeneity basis extrapolated worst in validation. Its error was
  largest at the origin with the longest relative extrapolation: a
  2024-07-01 cutoff, observed ages ≤ 5, forecasts to age 17.
- **Cash.** Tier upgrades, prices and the refund share are nearly stationary
  in tenure, so they extrapolate safely. The two calendar trends (cost and
  add-ons) are the riskiest assumption. The linear cost trend lowers
  2026 cost below the last observed level by $0.07 (January) to $0.80
  (December) per member-month, about 1% of contribution on average.
  Validation favoured including it (ablation below).

## 5. Chronological validation

**Design.** We used six historical origins with exclusive cutoffs on
2024-07-01, 2024-10-01, 2025-01-01, 2025-04-01, 2025-07-01 and 2025-10-01.
At each origin:

1. `panel.truncate_tables` recreates what a release at that cutoff would have
   contained: renewals due before the cutoff, and ledger rows, events,
   feedback, credit and spend dated before it.
2. The forecast grid is rebuilt from the truncated history with the lab's
   definition: cohorts acquired by the origin, twelve future months, capped
   at December 2025.
3. Every model, including the reference refit with the released
   `cohort_lab.model` code, is fitted only on the truncated data.
4. Outcomes come from later *released* months.

No 2026 outcome exists in this repository or was used. The score is the
contract's: contribution WAPE after summing channel/need cells within each
cohort-month. Model choice used the five origins with at least six future
months; the 2025-10-01 origin (3 months) is reported for completeness.

**How the origins resemble the final task.**

| Cutoff | Cohorts | Horizon | Oldest observed age | Forecast ages | Points |
|---|---:|---:|---:|---|---:|
| 2024-07-01 | 6 | 12 | 5 | 1–17 | 72 |
| 2024-10-01 | 9 | 12 | 8 | 1–20 | 108 |
| 2025-01-01 | 12 | 12 | 11 | 1–23 | 144 |
| 2025-04-01 | 15 | 9 | 14 | 1–23 | 135 |
| 2025-07-01 | 18 | 6 | 17 | 1–23 | 108 |
| 2025-10-01 | 21 | 3 | 20 | 1–23 | 63 |
| **Final task** | **24** | **12** | **23** | **1–35** | **288** |

The 12-month origins match the final horizon but have shallower histories.
The later origins match the final maturity but have shorter horizons. **No
origin tests forecasts beyond age 23**, so the age 24–35 extrapolation rests
on the assumptions in §4 rather than on direct evidence.

**Results: cohort-month contribution WAPE (lower is better).** The table
comes from `validation/results/backtest_by_origin.csv`.

| Model | 2024-07 | 2024-10 | 2025-01 | 2025-04 | 2025-07 | 2025-10 | Mean (selection) | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Selected: sBG ensemble + tier-aware cash** | **11.5%** | **5.9%** | **5.9%** | **4.1%** | **3.9%** | **3.8%** | **6.3%** | 11.5% |
| sBG cohort trend + tier-aware cash | 12.2% | 5.2% | 8.2% | 4.0% | 3.5% | 3.1% | 6.6% | 12.2% |
| sBG no cohort effect + tier-aware cash | 10.7% | 7.6% | 5.4% | 4.5% | 4.9% | 4.7% | 6.6% | 10.7% |
| BdW cohort trend + tier-aware cash | 16.6% | 4.3% | 6.8% | 4.3% | 3.5% | 3.1% | 7.1% | 16.6% |
| sBG random-walk cohort effects + tier-aware cash | 12.7% | 9.1% | 7.8% | 5.6% | 4.7% | 2.8% | 8.0% | 12.7% |
| **Reference: hierarchical sBG + Gamma (released, refit)** | 16.5% | 8.8% | 6.3% | 7.3% | 7.1% | 6.6% | 9.2% | 16.5% |
| Logit hazard + tier-aware cash | 22.8% | 7.4% | 7.7% | 5.3% | 4.4% | 3.1% | 9.5% | 22.8% |
| Template constant-hazard example | 45.5% | 42.6% | 37.9% | 32.2% | 27.0% | 20.7% | 37.0% | 45.5% |

Signed portfolio bias, selected vs reference:

| Cutoff | Selected | Reference |
|---|---:|---:|
| 2024-07 | −9.8% | −16.4% |
| 2024-10 | −0.3% | −8.3% |
| 2025-01 | +4.4% | −1.9% |
| 2025-04 | +0.2% | −5.8% |
| 2025-07 | −2.1% | −6.5% |
| 2025-10 | −3.3% | −6.5% |

The selected model's mean absolute bias is 3.3%, versus 7.8% for the
reference.

**Where the gain comes from.** Backtest totals for the sBG-trend member
versus the reference:

| Cutoff | Actual contribution per retained member | Ours | Reference |
|---|---:|---:|---:|
| 2024-07 | $30.75 | $30.44 | $27.99 |
| 2024-10 | $32.11 | $32.11 | $29.22 |
| 2025-01 | $33.44 | $33.42 | $30.70 |
| 2025-04 | $33.83 | $33.69 | $31.47 |
| 2025-07 | $33.95 | $34.02 | $32.01 |

The cash block is accurate to within about 1%. The remaining error is
retention, where the two approaches are similar: mean cell retention WAPE
is 9.3% for ours versus 8.9% for the reference. **The reference's Bayesian
retention is slightly better.** At the 2025-01 origin, both over-forecast
survival through 2025 for most 2024 cohorts, by roughly 2–9%.

**Ablations of the selected model** (mean selection WAPE):

| Ablation | Mean WAPE |
|---|---:|
| Full model | 6.3% |
| Flat (last-month) variable costs | 6.5% |
| No add-on calendar or price trend | 6.5% |
| No survivor tier mix (cell's historical renewal plan mix instead) | 14.2% |

**Caveats we want to be explicit about.**

1. The ensemble was chosen after comparing about eight variants on the same
   five origins, so its validation score is somewhat optimistic. The
   ranking of the *cash block* versus the reference is robust: every sBG
   or BdW variant with it beats the reference on average, and the effect is
   a consistent 6–9% per-member margin error. Paired with a logit hazard
   it does not (9.5%), so retention form still matters. The choice among sBG retention
   variants is less certain.
2. Origins overlap in calendar time, so they are not independent.
3. Format validity and backtest accuracy are not private-holdout accuracy.
4. We report expected values. Absolute-error scoring does not assess
   uncertainty calibration, and this deterministic model reports no
   intervals.

## 6. Information boundary

`predict(tables, prediction_grid, *, seed)` receives only the eight observed
tables and the public grid. It uses:

- people: stated need;
- acquisition attribution and campaign spend: channel;
- subscription periods: renewal outcomes due before the cutoff, plan tier;
- transactions: prices, add-ons, refunds and variable cost.

Pending renewals, product events, feedback, demographics and channel
holdouts are not used. The first three groups were tested and found
uninformative; holdouts are acquisition-only. No outcome after December
2025, simulator parameter or private target was searched for or used. The
code asserts that the December 2025 survivors it reconstructs equal the
grid's `origin_retained` in every cell.

## Reproducibility evidence

**Clean runs.** Two fresh clones were created in separate folders. Each ran
`uv sync --frozen` and then the standard command with `--seed 20261116`.
The two prediction files were byte-identical.

**Validator.** The output passed the instructor's public validator,
`lab.py validate-submission`, from the released lab at commit
`4848f4b75628bde8a43858bb54ca7c09de603f61`, after rebuilding its verified
SQL views in a scratch output folder.

**Visual checks:**
- all 1,152 grid rows are present;
- keys and exposures are unchanged;
- membership is nonincreasing and never exceeds origin survivors;
- contribution is at most net revenue;
- all values are finite and nonnegative.

**Evidence per commit.** The evidence for each submitted commit (prediction
SHA-256, runtime and environment) is in
`validation/results/reproducibility.json`.

**Re-running validation.**

```text
uv run --frozen python validation/backtest.py --data-dir /path/to/released/data     # our models (~1 min)
uv run --frozen python validation/fit_report.py --data-dir /path/to/released/data   # fitted settings
# reference refits need the released lab environment:
cd customer-economics-lab && uv run --frozen python ../<team-repo>/validation/reference_backtest.py --data-dir data --out ../<team-repo>/validation/results/reference
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Standard entry point. Exposes `predict(...)`, loads inputs, validates, writes predictions + receipt |
| `model.py` | Retention (sBG / BdW / logit) and tier-aware cash blocks; `SELECTED` ensemble |
| `panel.py` | Canonical cohort definitions, historical truncation, grid construction |
| `pyproject.toml`, `uv.lock` | Locked Python 3.12 environment (pandas, numpy, scipy) |
| `submission-contract.json`, `prediction_grid.csv` | Unmodified challenge contract and public grid |
| `validation/` | Origins, backtest, reference refit scripts, results and fitted settings |
| `scaffold-receipt.json` | Hashes of the released scaffold this project started from |

## References

- Fader, P. S., & Hardie, B. G. S. (2007). How to project customer retention. *Journal of Interactive Marketing*, 21(1), 76–90.
- Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J., & Steenburgh, T. (2018). "How to project customer retention" revisited: The role of duration dependence. *Journal of Interactive Marketing*, 43, 1–16.
- Fader, P. S., & Hardie, B. G. S. (2010). Customer-base valuation in a contractual setting: The perils of ignoring heterogeneity. *Marketing Science*, 29(1), 85–93.
- Gupta, S., Lehmann, D. R., & Stuart, J. A. (2004). Valuing customers. *Journal of Marketing Research*, 41(1), 7–18.
- McCarthy, D. M., Fader, P. S., & Hardie, B. G. S. (2017). Valuing subscription-based businesses using publicly disclosed customer data. *Journal of Marketing*, 81(1), 17–35.
- Orduz, J. C. (2024). Cohort revenue retention analysis with Flax and NumPyro (adapted by the released reference).
- Studio North customer-economics lab (released reference model, data and challenge contract).
