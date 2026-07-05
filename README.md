# Experimental

Experimental is a Streamlit app for running an end-to-end A/B test readout from
user-level experiment data. It covers the parts of experimentation that usually
get scattered across notebooks: power planning, assignment diagnostics,
frequentist and Bayesian analysis, sequential monitoring, segment checks, and a
final launch recommendation.

The demo project uses a new-user onboarding checklist experiment. The included
sample data produces a `SHIP WITH MONITORING` recommendation: treatment improves
day-7 retention by 1.55 percentage points, assignment diagnostics look clean,
and guardrails do not show a major regression.

## What The App Covers

- Power analysis and minimum detectable effect planning
- CSV upload and column mapping for user-level experiment data
- Sample ratio mismatch checks and A/A randomization simulation
- Two-proportion z-test with confidence intervals for binary metrics
- Beta-binomial Bayesian readout for the primary metric
- Guardrail monitoring for conversion and continuous metrics
- O'Brien-Fleming sequential testing boundary checks
- Segment-level lift analysis for heterogeneous treatment effects
- Decision summary page for ship, monitor, or hold recommendations

## Pages

1. Home - overview of the experimentation workflow
2. Power Analysis - sample size and sensitivity planning
3. Data and Setup - demo loader, CSV upload, and column configuration
4. Data Validation - SRM and A/A checks before reading results
5. Statistical Analysis - primary metric, guardrails, and Bayesian readout
6. Sequential Testing - planned-look boundary evaluation
7. Segment Analysis - lift by country, platform, or user segment
8. Decision Summary - launch recommendation and supporting evidence

## Quick Start

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL and choose **Load demo dataset** on the Data and
Setup page.

To run the test suite:

```bash
python3 -m unittest discover tests
```

## Demo Data

The committed demo dataset is at:

```text
data/user_metrics.csv
```

It contains one row per assigned user, including:

- `user_id`
- `variant`
- `day_7_retained`
- `revenue_7d`
- `crashed`
- `p95_load_time_ms`
- `country`
- `platform`
- `user_segment`

The synthetic data generation code lives in `generate_data.py` and the
supporting pipeline modules under `src/`.

## Repo Layout

```text
app.py                         Streamlit app and eight-page workflow
data/user_metrics.csv          Demo user-level experiment dataset
generate_data.py               Synthetic experiment-data generator
generate_figures.py            Figure generation for the markdown report
generate_report.py             Markdown decision memo generator
reports/                       Generated experiment decision report
src/analysis.py                Frequentist binary metric analysis
src/bayesian.py                Beta-binomial posterior summary
src/monitoring.py              Peeking simulation and sequential boundaries
src/power.py                   Sample-size and power calculations
src/quality_checks.py          SRM and A/A diagnostic checks
src/segments.py                Segment-level treatment effects
tests/test_pipeline.py         Smoke tests for assignment and analysis flow
```

## Implementation Notes

The Streamlit layer keeps the workflow in one file so the page flow is easy to
review. The reusable experiment logic is split into small modules under `src/`,
which makes the statistical checks testable without the UI.

The app is intended as a product data science portfolio project rather than a
replacement for a production experimentation platform. In a production setting,
the same analysis layer would usually sit behind stronger metric definitions,
warehouse-backed data contracts, experiment registry metadata, and automated
monitoring jobs.
