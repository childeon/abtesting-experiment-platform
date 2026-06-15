import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as scipy_beta
from scipy.stats import norm as scipy_norm
from scipy.stats import ttest_ind
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.analysis import analyze_binary_metric
from src.bayesian import bayesian_binary_metric_readout
from src.monitoring import evaluate_sequential_look, obrien_fleming_boundary
from src.power import calculate_sample_size
from src.quality_checks import check_sample_ratio_mismatch, run_aa_test
from src.segments import analyze_segment_effects


DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "user_metrics.csv")
COLOR_CONTROL = "#94A3B8"
COLOR_TREATMENT = "#818CF8"
COLOR_POSITIVE = "#34D399"
COLOR_NEGATIVE = "#F87171"
COLOR_NEUTRAL = "#64748B"
COLOR_AMBER = "#FBBF24"
PT = "plotly_dark"

st.set_page_config(
    page_title="Experimental · A/B Testing Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0F172A; }
[data-testid="stSidebar"] {
    background: #1E293B;
    border-right: 1px solid #334155;
}
.hero {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 24px;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0 0 8px; color: #F1F5F9; }
.hero p  { font-size: 1.05rem; color: #94A3B8; margin: 0; }
.decision-banner {
    border-radius: 16px;
    padding: 32px 40px;
    text-align: center;
    margin-bottom: 24px;
}
.badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-pass   { background: #064e3b; color: #6ee7b7; }
.badge-fail   { background: #7f1d1d; color: #fca5a5; }
.badge-warn   { background: #78350f; color: #fcd34d; }
.badge-neutral{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
</style>
""",
    unsafe_allow_html=True,
)


_DEFAULTS = {
    "data": None,
    "variant_col": "variant",
    "control_val": "control",
    "treatment_val": "treatment",
    "primary_metric": None,
    "guardrail_metrics": [],
    "segment_cols": [],
    "alpha": 0.05,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def pct(v, d=2):
    return f"{v * 100:.{d}f}%"


def lift(v, d=2):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{d}f}pp"


def badge(text, kind="pass"):
    return f'<span class="badge badge-{kind}">{text}</span>'


@st.cache_data
def demo_row_count(path):
    return len(pd.read_csv(path, usecols=["user_id"]))


def data_ready():
    return (
        st.session_state.data is not None
        and st.session_state.primary_metric is not None
    )


def chart_defaults():
    return dict(
        template=PT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


with st.sidebar:
    st.markdown("## Experimental")
    st.caption("A/B Testing Analysis Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Power Analysis",
            "Data and Setup",
            "Data Validation",
            "Statistical Analysis",
            "Sequential Testing",
            "Segment Analysis",
            "Decision Summary",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    if data_ready():
        df_sb = st.session_state.data
        st.success("Data loaded")
        st.caption(f"**Primary:** `{st.session_state.primary_metric}`")
        ctrl_n = int((df_sb[st.session_state.variant_col] == st.session_state.control_val).sum())
        trt_n = int((df_sb[st.session_state.variant_col] == st.session_state.treatment_val).sum())
        st.caption(f"Control {ctrl_n:,}   Treatment {trt_n:,}")
    else:
        st.info("Load data on the **Data and Setup** page.")

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Experimental</h1>
            <p>An end-to-end A/B testing analysis platform for product data scientists.<br>
            Design experiments, validate data quality, analyze results, and make ship decisions. All in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Experiment Design**")
        st.markdown(
            "Power analysis and sample size calculation with sensitivity curves. "
            "Know your required sample before you ship a single user."
        )
    with c2:
        st.markdown("**Data Validation**")
        st.markdown(
            "Sample ratio mismatch detection (chi-square) and A/A test simulation "
            "to verify your randomization infrastructure is sound."
        )
    with c3:
        st.markdown("**Statistical Analysis**")
        st.markdown(
            "Frequentist (z-test + CIs) and Bayesian (Beta-Binomial posteriors) "
            "analysis side by side. Guardrail metric monitoring included."
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("**Sequential Testing**")
        st.markdown(
            "O'Brien-Fleming boundaries tell you the z-statistic threshold "
            "required to stop early at each planned look without inflating your false positive rate."
        )
    with c5:
        st.markdown("**Segment Analysis**")
        st.markdown(
            "Break down treatment effects by country, platform, user segment, "
            "or any categorical feature. Catch effects hidden by the average."
        )
    with c6:
        st.markdown("**Decision Summary**")
        st.markdown(
            "Automated ship / hold / no-ship recommendation synthesizing the "
            "primary metric, guardrails, and Bayesian corroboration."
        )

    st.divider()

    st.markdown("### How to use this platform")
    st.markdown(
        """
1. **Power Analysis** → design the experiment before running it
2. **Data and Setup** → upload your user-level CSV (or load the demo dataset)
3. **Data Validation** → check for SRM and verify randomization health
4. **Statistical Analysis** → frequentist + Bayesian readout on your primary metric
5. **Sequential Testing** → evaluate whether you can stop the experiment early
6. **Segment Analysis** → inspect heterogeneous treatment effects
7. **Decision Summary** → get a synthesis with a ship recommendation
        """
    )

    with st.expander("What is A/B testing?"):
        st.markdown(
            """
**A/B testing** (randomized controlled experiment) shows two versions of an experience
to randomly assigned user groups, then measures which produces a better outcome.

**Why randomization matters** Random assignment ensures the only *systematic* difference
between groups is the treatment, which makes causal inference valid rather than correlational.

**The core tension** Small effects need large samples to detect reliably. Power analysis
quantifies that trade-off before you run the experiment.

**The decision framework used here**
- Primary metric must improve significantly (p < α)
- No guardrail metrics can significantly degrade
- Bayesian P(B > A) ≥ 95% provides a second signal
            """
        )

elif page == "Power Analysis":
    st.title("Power Analysis")
    st.caption("Design your experiment before running it. Calculate required sample size and runtime.")

    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.subheader("Parameters")

        baseline_rate = st.number_input(
            "Baseline conversion rate",
            min_value=0.001, max_value=0.998, value=0.30, step=0.01, format="%.3f",
            help=(
                "The current metric rate in the control group. "
                "E.g. if 30% of users retain on day 7, enter 0.30. "
                "Sets the starting point for effect size calculation. "
                "Variance of a proportion depends on its value, so the same absolute MDE "
                "is harder to detect near 0.5 than near 0.05."
            ),
        )

        max_mde = round(min(0.20, 0.999 - baseline_rate - 0.001), 3)
        mde = st.number_input(
            "Minimum Detectable Effect (MDE, absolute pp)",
            min_value=0.001, max_value=max_mde, value=min(0.03, max_mde), step=0.005, format="%.3f",
            help=(
                "The smallest absolute lift you need to reliably detect. "
                "If baseline is 30% and MDE is 3pp, you want to detect if treatment reaches 33%. "
                "Halving the MDE roughly quadruples the required sample size (1/MDE² relationship). "
                "Set this to the minimum business-meaningful effect, not the effect you hope to see."
            ),
        )

        alpha = st.select_slider(
            "Significance level (α)",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            help=(
                "Probability of a false positive, meaning the test concludes an effect exists when it doesn't. "
                "α = 0.05 means a 5% false alarm rate. Lower α is stricter but requires more users. "
                "Industry default is 0.05. Use 0.01 for high-stakes decisions."
            ),
        )

        power = st.select_slider(
            "Statistical power (1 − β)",
            options=[0.70, 0.80, 0.90, 0.95],
            value=0.80,
            help=(
                "Probability of detecting a true effect of exactly MDE. "
                "Power = 0.80 means if the treatment truly lifts the metric by MDE, "
                "you'll detect it in 80% of runs. The other 20% are false negatives. "
                "Industry default is 0.80. High-stakes experiments use 0.90."
            ),
        )

        daily_users = st.number_input(
            "Daily eligible users (all variants combined)",
            min_value=10, max_value=10_000_000, value=500, step=50,
            help=(
                "New users who qualify to enter the experiment each day. "
                "Drives how long the experiment runs. "
                "Count only users who meet your eligibility criteria, not total DAU."
            ),
        )

        maturity_days = st.number_input(
            "Metric maturity window (days)",
            min_value=0, max_value=365, value=7, step=1,
            help=(
                "Days after enrollment needed to observe the metric. "
                "For day-7 retention this is 7, for immediate click-through this is 0. "
                "Added on top of enrollment time to get total calendar days."
            ),
        )

        n_variants = st.number_input(
            "Number of variants (including control)",
            min_value=2, max_value=10, value=2, step=1,
            help=(
                "Total arms in the experiment. A/B test = 2, A/B/C test = 3. "
                "More variants split traffic further and extend the experiment. "
                "α is Bonferroni-corrected to α / (n_variants − 1) per comparison."
            ),
        )

    with col_out:
        st.subheader("Results")
        try:
            res = calculate_sample_size(
                baseline_rate=baseline_rate,
                minimum_detectable_effect=mde,
                alpha=alpha,
                power=power,
                daily_eligible_users=daily_users,
                metric_maturity_days=maturity_days,
                n_variants=n_variants,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Users per variant",
                f"{res['n_per_variant']:,}",
                help="Required users in each arm to achieve the target power at the chosen MDE.",
            )
            m2.metric(
                "Total users",
                f"{res['total_n']:,}",
                help="n_per_variant × n_variants. Spread across all arms.",
            )
            m3.metric(
                "Calendar days",
                f"{res['total_calendar_days']:,}",
                help="Enrollment days + metric maturity window = total experiment runtime.",
            )

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "Enrollment days",
                f"{res['enrollment_days']:,}",
                help="Days to enroll total_n users at the given daily traffic rate.",
            )
            r2.metric(
                "Target rate",
                pct(res["target_rate"]),
                delta=lift(res["minimum_detectable_effect"]),
                help="Baseline + MDE. The treatment rate you're designing to detect.",
            )
            r3.metric(
                "Corrected α",
                f"{res['corrected_alpha']:.4f}",
                help=(
                    "Bonferroni-corrected threshold at α ÷ (n_variants − 1). "
                    "Controls family-wise error rate when testing multiple treatments."
                ),
            )

            with st.expander("How sample size is computed"):
                st.markdown(
                    f"""
**Cohen's h** converts your two proportions into a standardized effect size,
accounting for the fact that variance of a proportion depends on its value.

```
h = 2 arcsin(√p₁) − 2 arcsin(√p₀)
```

- Baseline p₀ = {pct(baseline_rate)}, Target p₁ = {pct(res['target_rate'])}
- Corrected α = {alpha} ÷ {n_variants - 1} = **{res['corrected_alpha']:.4f}**
- `NormalIndPower.solve_power(effect_size=h, alpha={res['corrected_alpha']:.4f}, power={power})`
gives **{res['n_per_variant']:,}** users per variant

Runtime is {res['enrollment_days']} enrollment days plus {maturity_days} maturity days
giving **{res['total_calendar_days']} calendar days** total.
                    """
                )

        except ValueError as e:
            st.error(f"Invalid parameters: {e}")
            res = None

    st.divider()
    st.subheader("Sensitivity Analysis")

    tab_mde, tab_pwr = st.tabs(["MDE vs Sample Size", "Power Curve"])

    with tab_mde:
        try:
            mde_vals = np.linspace(0.005, min(0.20, 0.999 - baseline_rate), 60)
            n_vals = []
            for m in mde_vals:
                try:
                    r = calculate_sample_size(
                        baseline_rate=baseline_rate,
                        minimum_detectable_effect=float(m),
                        alpha=alpha, power=power,
                        daily_eligible_users=daily_users,
                        metric_maturity_days=maturity_days,
                        n_variants=n_variants,
                    )
                    n_vals.append(r["n_per_variant"])
                except Exception:
                    n_vals.append(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[m * 100 for m in mde_vals], y=n_vals,
                mode="lines", line=dict(color=COLOR_TREATMENT, width=2.5),
                fill="tozeroy", fillcolor="rgba(129,140,248,0.1)",
            ))
            fig.add_vline(
                x=mde * 100, line_dash="dash", line_color=COLOR_NEGATIVE,
                annotation_text=f"Current MDE: {mde*100:.1f}pp",
                annotation_font_color=COLOR_NEGATIVE,
            )
            fig.update_layout(
                **chart_defaults(), xaxis_title="MDE (percentage points)",
                yaxis_title="Users per Variant", height=340, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Halving the MDE roughly quadruples the required sample. The curve is convex.")
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")

    with tab_pwr:
        try:
            corrected_a = alpha / max(n_variants - 1, 1)
            es = proportion_effectsize(baseline_rate + mde, baseline_rate)
            n_max = max(5000, (res["n_per_variant"] * 2 if res else 5000))
            n_range = np.linspace(50, n_max, 120)
            analysis_obj = NormalIndPower()
            pw_curve = []
            for n in n_range:
                try:
                    pw = analysis_obj.solve_power(
                        effect_size=es, nobs1=n,
                        alpha=corrected_a, ratio=1.0, alternative="two-sided",
                    )
                    pw_curve.append(min(float(pw), 1.0))
                except Exception:
                    pw_curve.append(None)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=n_range, y=pw_curve,
                mode="lines", line=dict(color=COLOR_POSITIVE, width=2.5),
            ))
            fig2.add_hline(
                y=power, line_dash="dash", line_color=COLOR_AMBER,
                annotation_text=f"Target power: {power:.0%}",
                annotation_font_color=COLOR_AMBER,
            )
            if res:
                fig2.add_vline(
                    x=res["n_per_variant"], line_dash="dash", line_color=COLOR_TREATMENT,
                    annotation_text=f"Required n: {res['n_per_variant']:,}",
                    annotation_font_color=COLOR_TREATMENT,
                )
            fig2.update_layout(
                **chart_defaults(), xaxis_title="Users per Variant",
                yaxis_title="Power", yaxis_tickformat=".0%",
                height=340, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "The curve flattens out. Going from 90% to 95% power costs far more users "
                "than going from 70% to 80%."
            )
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")

elif page == "Data and Setup":
    st.title("Data and Setup")
    st.caption("Upload your experiment dataset and map columns to roles.")

    with st.expander("What data format is expected?"):
        st.markdown(
            """
Upload a **user-level CSV** where each row is one user and columns are pre-computed metrics.

**Required**
- A **variant column** with values identifying control vs treatment
- At least one **binary metric column** (0 or 1) as the primary metric

**Optional**
- Guardrail metrics (binary or continuous)
- Segment columns (categorical) for heterogeneous effect analysis

Metrics must already be aggregated at the user level. The platform does not compute
metrics from raw event logs.
            """
        )

    tab_upload, tab_demo = st.tabs(["Upload CSV", "Demo Dataset"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload user-level experiment CSV",
            type=["csv"],
            help="Each row = one user. Columns = variant label + pre-computed metric values.",
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.data = df
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    with tab_demo:
        demo_n_users = demo_row_count(DEMO_DATA_PATH)
        st.markdown(
            f"Simulated dataset with **{demo_n_users:,} users** from a new-user onboarding checklist A/B test. "
            "Primary metric is `day_7_retained` (binary) and segments include country, platform, and user_segment."
        )
        if st.button("Load Demo Data", type="primary"):
            df = pd.read_csv(DEMO_DATA_PATH)
            st.session_state.data = df
            st.session_state.variant_col = "variant"
            st.session_state.control_val = "control"
            st.session_state.treatment_val = "treatment"
            st.session_state.primary_metric = "day_7_retained"
            st.session_state.guardrail_metrics = ["crashes_first_7_days"]
            st.session_state.segment_cols = ["country", "platform", "user_segment"]
            st.success("Demo data loaded and configuration pre-filled.")

    if st.session_state.data is None:
        st.stop()

    df = st.session_state.data
    all_cols = df.columns.tolist()

    st.divider()
    st.subheader("Column Configuration")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        variant_col = st.selectbox(
            "Variant column",
            all_cols,
            index=all_cols.index(st.session_state.variant_col)
            if st.session_state.variant_col in all_cols else 0,
            help="Column identifying which variant each user was assigned to.",
        )
        st.session_state.variant_col = variant_col

        unique_variants = df[variant_col].dropna().unique().tolist() if variant_col else []
        control_val = st.selectbox(
            "Control variant label",
            unique_variants,
            index=unique_variants.index(st.session_state.control_val)
            if st.session_state.control_val in unique_variants else 0,
            help="The variant column value that identifies the control (baseline) group.",
        )
        st.session_state.control_val = control_val

        remaining = [v for v in unique_variants if v != control_val]
        treatment_val = st.selectbox(
            "Treatment variant label",
            remaining,
            index=remaining.index(st.session_state.treatment_val)
            if st.session_state.treatment_val in remaining else 0,
            help="The variant column value that identifies the treatment group.",
        )
        st.session_state.treatment_val = treatment_val

    with col_b:
        binary_cols = [
            c for c in all_cols
            if c != variant_col and df[c].dropna().isin([0, 1]).all()
        ]
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != variant_col
        ]
        primary_opts = binary_cols if binary_cols else numeric_cols

        primary_metric = st.selectbox(
            "Primary metric (binary 0/1)",
            primary_opts,
            index=primary_opts.index(st.session_state.primary_metric)
            if st.session_state.primary_metric in primary_opts else 0,
            help=(
                "The single metric your launch decision hinges on. "
                "Must be binary where 0 means not observed and 1 means it was. "
                "Examples include day_7_retained, converted, and activated."
            ),
        )
        st.session_state.primary_metric = primary_metric

        guardrail_opts = [c for c in numeric_cols if c != primary_metric]
        guardrail_metrics = st.multiselect(
            "Guardrail metrics",
            guardrail_opts,
            default=[m for m in st.session_state.guardrail_metrics if m in guardrail_opts],
            help=(
                "Metrics you must NOT degrade. A significant guardrail breach blocks "
                "the ship even if the primary metric wins. "
                "Examples include crash_rate, revenue_per_user, and p95_load_time_ms."
            ),
        )
        st.session_state.guardrail_metrics = guardrail_metrics

        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != variant_col
        ]
        segment_cols = st.multiselect(
            "Segment columns",
            cat_cols,
            default=[c for c in st.session_state.segment_cols if c in cat_cols],
            help=(
                "Categorical columns used for heterogeneous treatment effect analysis. "
                "Examples include country, platform, user_segment, and device_type."
            ),
        )
        st.session_state.segment_cols = segment_cols

    alpha_val = st.select_slider(
        "Global significance level (α)",
        options=[0.01, 0.05, 0.10],
        value=st.session_state.alpha,
        help="Significance threshold applied across all statistical tests on this platform.",
    )
    st.session_state.alpha = alpha_val

    st.divider()
    st.subheader("Data Preview")

    c1, c2, c3, c4 = st.columns(4)
    ctrl_n = int((df[variant_col] == control_val).sum())
    trt_n = int((df[variant_col] == treatment_val).sum())
    c1.metric("Total Users", f"{len(df):,}")
    c2.metric(f"Control ({control_val})", f"{ctrl_n:,}")
    c3.metric(f"Treatment ({treatment_val})", f"{trt_n:,}")
    c4.metric("Columns", f"{len(df.columns)}")

    preview_cols = [variant_col, primary_metric] + guardrail_metrics[:2] + segment_cols[:2]
    preview_cols = [c for c in preview_cols if c in df.columns]
    st.dataframe(df[preview_cols].head(20), use_container_width=True)

elif page == "Data Validation":
    st.title("Data Validation")
    st.caption("Verify data integrity before interpreting results. Always run these checks first.")

    if not data_ready():
        st.warning("Load your data on the **Data and Setup** page first.")
        st.stop()

    df = st.session_state.data
    variant_col = st.session_state.variant_col
    control_val = st.session_state.control_val
    treatment_val = st.session_state.treatment_val
    alpha = st.session_state.alpha
    primary_metric = st.session_state.primary_metric

    tab_srm, tab_aa = st.tabs(["Sample Ratio Mismatch", "A/A Test Simulation"])

    with tab_srm:
        st.subheader("Sample Ratio Mismatch (SRM)")

        with st.expander("What is SRM and why does it matter?"):
            st.markdown(
                """
**Sample Ratio Mismatch** occurs when the actual split of users differs from the
intended traffic allocation, e.g. you targeted 50/50 but observed 54/46.

**Why it's dangerous** If the assignment mechanism is broken, groups may differ
systematically on unobserved characteristics and all effect estimates become biased.
A detected SRM invalidates the experiment.

**Detection** A chi-square goodness-of-fit test compares observed counts to expected
counts under the intended allocation. SRM tests conventionally use α = 0.001,
much stricter than the experiment α because a mismatch is a data quality problem,
not a business signal.

**If detected** Stop the analysis. Investigate assignment logic, event logging,
and data pipelines before proceeding.
                """
            )

        col_cfg, col_res = st.columns(2, gap="large")

        with col_cfg:
            ctrl_alloc = st.number_input(
                f"Expected {control_val} share",
                min_value=0.01, max_value=0.99, value=0.50, step=0.05, format="%.2f",
                help="Intended fraction of traffic assigned to control.",
            )
            trt_alloc = round(1.0 - ctrl_alloc, 4)
            st.markdown(f"Expected `{treatment_val}` share is **{trt_alloc:.2f}** (auto)")

            srm_alpha = st.number_input(
                "SRM α threshold",
                min_value=0.0001, max_value=0.05, value=0.001, step=0.0005, format="%.4f",
                help=(
                    "Significance threshold for the SRM test. Use 0.001 rather than 0.05. "
                    "A mismatch at p < 0.001 is a strong signal of infrastructure breakage."
                ),
            )

        with col_res:
            try:
                assignments = df[[variant_col]].rename(columns={variant_col: "variant"})
                srm = check_sample_ratio_mismatch(
                    assignments=assignments,
                    traffic_allocation={control_val: ctrl_alloc, treatment_val: trt_alloc},
                    alpha=srm_alpha,
                )

                obs = srm["observed_counts"]
                exp = srm["expected_counts"]

                for var in [control_val, treatment_val]:
                    o = obs.get(var, 0)
                    e = exp.get(var, 0)
                    delta_pct = (o - e) / e * 100 if e > 0 else float("nan")
                    st.markdown(
                        f"`{var}` observed **{o:,}**, expected **{int(e):,}** "
                        f"({delta_pct:+.1f}%)"
                    )

                st.markdown(f"χ² = `{srm['chi_square_statistic']:.4f}` | p = `{srm['p_value']:.5f}`")

                if srm["srm_detected"]:
                    st.markdown(badge("SRM DETECTED, do not ship", "fail"), unsafe_allow_html=True)
                    st.error(
                        "Observed allocation deviates significantly from expected. "
                        "Investigate your assignment pipeline before interpreting results."
                    )
                else:
                    st.markdown(badge("No SRM detected", "pass"), unsafe_allow_html=True)
                    st.success("Assignment ratio is consistent with intended allocation.")

                # Bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Observed",
                    x=[control_val, treatment_val],
                    y=[obs.get(control_val, 0), obs.get(treatment_val, 0)],
                    marker_color=COLOR_TREATMENT,
                ))
                fig.add_trace(go.Bar(
                    name="Expected",
                    x=[control_val, treatment_val],
                    y=[int(exp.get(control_val, 0)), int(exp.get(treatment_val, 0))],
                    marker_color=COLOR_CONTROL,
                    marker_pattern_shape="x",
                ))
                fig.update_layout(
                    **chart_defaults(), barmode="group",
                    xaxis_title="Variant", yaxis_title="User Count",
                    height=280, legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.error(f"SRM check failed: {exc}")

    with tab_aa:
        st.subheader("A/A Test Simulation")

        with st.expander("What is an A/A test?"):
            st.markdown(
                """
An **A/A test** applies your experiment's statistical test to two groups that received
**identical** experiences with no treatment at all.

**Why run it** Your infrastructure is healthy if and only if the A/A false positive rate
approximates the nominal α. If you see p < 0.05 in 25% of A/A runs when α = 0.05,
something is wrong, whether that's biased randomization, metric computation error,
or a variance estimation issue.

**This simulation** draws synthetic user outcomes from the same distribution in both
arms and counts how often the z-test incorrectly fires. The observed rate should
land within a few percentage points of α.
                """
            )

        col_ai, col_ao = st.columns(2, gap="large")

        with col_ai:
            data_baseline = float(df[primary_metric].mean()) if primary_metric in df.columns else 0.30
            aa_n = st.number_input(
                "Users per simulation run",
                min_value=100, max_value=100_000,
                value=min(len(df), 2000), step=100,
                help="Sample size per simulated A/A experiment.",
            )
            aa_baseline = st.number_input(
                "Baseline rate",
                min_value=0.01, max_value=0.99,
                value=round(data_baseline, 3), step=0.01,
                help=f"Detected from your data at {data_baseline:.3f}. Controls the simulation.",
            )
            aa_sims = st.number_input(
                "Simulations",
                min_value=100, max_value=5000, value=1000, step=100,
                help="More simulations give a more stable FPR estimate. 1000 is sufficient.",
            )
            run_aa = st.button("Run A/A Simulation", type="primary", use_container_width=True)

        with col_ao:
            if run_aa:
                with st.spinner("Running simulations…"):
                    aa_res = run_aa_test(
                        n_users=aa_n,
                        baseline_rate=aa_baseline,
                        alpha=alpha,
                        n_simulations=aa_sims,
                    )
                fpr = aa_res["false_positive_rate"]
                tol = 0.02
                st.metric(
                    "Observed false positive rate",
                    f"{fpr:.1%}",
                    delta=f"{(fpr - alpha)*100:+.1f}pp vs expected {alpha:.0%}",
                    delta_color="inverse" if fpr > alpha + tol else "off",
                    help="Fraction of simulated A/A tests that incorrectly rejected H₀.",
                )

                if abs(fpr - alpha) <= tol:
                    st.markdown(badge("FPR within expected range", "pass"), unsafe_allow_html=True)
                    st.success(
                        f"FPR ({fpr:.1%}) is close to α ({alpha:.0%}). "
                        "Randomization infrastructure looks healthy."
                    )
                else:
                    st.markdown(badge("FPR outside expected range", "warn"), unsafe_allow_html=True)
                    st.warning(
                        f"FPR ({fpr:.1%}) deviates from α ({alpha:.0%}). "
                        "Investigate metric computation or assignment logic."
                    )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fpr * 100,
                    number={"suffix": "%", "font": {"size": 32}},
                    title={"text": "False Positive Rate"},
                    gauge={
                        "axis": {"range": [0, 20], "ticksuffix": "%"},
                        "bar": {"color": COLOR_TREATMENT},
                        "steps": [
                            {"range": [0, 7], "color": "#064e3b"},
                            {"range": [7, 12], "color": "#78350f"},
                            {"range": [12, 20], "color": "#7f1d1d"},
                        ],
                        "threshold": {
                            "line": {"color": COLOR_AMBER, "width": 3},
                            "thickness": 0.75,
                            "value": alpha * 100,
                        },
                    },
                ))
                fig.update_layout(
                    **chart_defaults(), height=240,
                    margin=dict(l=20, r=20, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Yellow line = nominal α ({alpha:.0%}). Green zone = healthy.")

elif page == "Statistical Analysis":
    st.title("Statistical Analysis")
    st.caption("Frequentist and Bayesian analysis of your experiment results.")

    if not data_ready():
        st.warning("Load your data on the **Data and Setup** page first.")
        st.stop()

    df = st.session_state.data
    variant_col = st.session_state.variant_col
    control_val = st.session_state.control_val
    treatment_val = st.session_state.treatment_val
    primary_metric = st.session_state.primary_metric
    guardrail_metrics = st.session_state.guardrail_metrics
    alpha = st.session_state.alpha

    freq_tab, bayes_tab = st.tabs(["Frequentist", "Bayesian"])

    with freq_tab:
        st.subheader("Frequentist Analysis")

        with st.expander("How the frequentist test works"):
            st.markdown(
                f"""
**Two-proportion z-test** comparing treatment vs control on the primary binary metric.

- **H₀** treatment rate = control rate
- **H₁** treatment rate ≠ control rate (two-sided, because we care about harms too)
- Reject H₀ if p < α = {alpha}

**Confidence interval** (Wald method)
`lift ± z_{{α/2}} × √( p̂₁(1−p̂₁)/n₁ + p̂₀(1−p̂₀)/n₀ )`

**What the CI means** If we ran this experiment many times, {(1-alpha):.0%} of the
computed intervals would contain the true effect. It does *not* mean there is a
{(1-alpha):.0%} chance the true effect is in this specific interval. That is Bayesian thinking.
                """
            )

        try:
            freq = analyze_binary_metric(
                user_metrics=df, metric_col=primary_metric,
                variant_col=variant_col,
                control_variant=control_val, treatment_variant=treatment_val,
                alpha=alpha,
            )

            st.markdown(f"**Primary metric** `{primary_metric}`")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                f"Control rate",
                pct(freq["control_rate"]),
                help=f"Observed rate in {control_val}. n = {freq['n_control']:,}.",
            )
            m2.metric(
                f"Treatment rate",
                pct(freq["treatment_rate"]),
                help=f"Observed rate in {treatment_val}. n = {freq['n_treatment']:,}.",
            )
            m3.metric(
                "Absolute lift",
                lift(freq["absolute_lift"]),
                help="Treatment rate − control rate (percentage points).",
            )
            m4.metric(
                "Relative lift",
                f"{freq['relative_lift_pct']:+.1f}%",
                help="Absolute lift ÷ control rate. More interpretable for stakeholders.",
            )

            s1, s2, s3 = st.columns(3)
            s1.metric(
                "z-statistic",
                f"{freq['z_statistic']:.3f}",
                help="Standardized test statistic. Significant if |z| > 1.96 (for α=0.05).",
            )
            s2.metric(
                "p-value",
                f"{freq['p_value']:.4f}",
                help=f"Probability of observing this lift or larger if H₀ is true. Significant if < {alpha}.",
            )
            sig_label = "SIGNIFICANT" if freq["significant"] else "NOT SIGNIFICANT"
            sig_kind = "pass" if freq["significant"] and freq["absolute_lift"] > 0 else (
                "fail" if freq["significant"] else "neutral"
            )
            s3.markdown("**Status**")
            s3.markdown(badge(sig_label, sig_kind), unsafe_allow_html=True)

            ci_lo, ci_hi = freq["ci_95_absolute_lift"]

            fig = go.Figure()
            fig.add_shape(
                type="line", x0=0, x1=0, y0=-0.5, y1=1.5,
                line=dict(color=COLOR_NEUTRAL, width=1, dash="dash"),
            )
            dot_color = COLOR_POSITIVE if freq["absolute_lift"] > 0 else COLOR_NEGATIVE
            fig.add_trace(go.Scatter(
                x=[freq["absolute_lift"] * 100], y=[0],
                mode="markers",
                marker=dict(color=dot_color, size=14, symbol="diamond"),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[(ci_hi - freq["absolute_lift"]) * 100],
                    arrayminus=[(freq["absolute_lift"] - ci_lo) * 100],
                    color=COLOR_CONTROL, thickness=2.5, width=10,
                ),
                hovertemplate=(
                    f"<b>{primary_metric}</b><br>"
                    f"Lift: {freq['absolute_lift']*100:+.2f}pp<br>"
                    f"{(1-alpha):.0%} CI: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]pp<br>"
                    f"p = {freq['p_value']:.4f}<extra></extra>"
                ),
                showlegend=False,
            ))
            fig.update_layout(
                **chart_defaults(),
                xaxis_title="Absolute Lift (percentage points)",
                yaxis_visible=False, height=160,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Diamond = point estimate. Whiskers = {(1-alpha):.0%} CI "
                f"[{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp. "
                f"Dashed line = null hypothesis (zero effect)."
            )

        except Exception as exc:
            st.error(f"Frequentist analysis failed: {exc}")
            freq = None

        if guardrail_metrics:
            st.divider()
            st.subheader("Guardrail Metrics")

            with st.expander("What are guardrail metrics?"):
                st.markdown(
                    """
**Guardrail metrics** are metrics you must not degrade, even if the primary metric improves.

They protect against unintended side effects.
- A faster page might boost retention but increase crash rate
- A recommendation change might lift clicks but hurt revenue

**Decision rule** Ship only if primary metric wins **AND** no guardrail is significantly breached.

Binary guardrails use a z-test. Continuous guardrails use a two-sample t-test.
                    """
                )

            guard_rows = []
            for gm in guardrail_metrics:
                if gm not in df.columns:
                    continue
                is_bin = df[gm].dropna().isin([0, 1]).all()
                ctrl_s = df[df[variant_col] == control_val][gm].dropna()
                trt_s = df[df[variant_col] == treatment_val][gm].dropna()
                ctrl_m = ctrl_s.mean()
                trt_m = trt_s.mean()
                delta = trt_m - ctrl_m
                rel = delta / ctrl_m * 100 if ctrl_m != 0 else float("nan")

                if is_bin:
                    try:
                        gr = analyze_binary_metric(
                            df, metric_col=gm, variant_col=variant_col,
                            control_variant=control_val, treatment_variant=treatment_val,
                            alpha=alpha,
                        )
                        p_val = gr["p_value"]
                        sig = gr["significant"]
                    except Exception:
                        p_val = float("nan")
                        sig = False
                else:
                    _, p_val = ttest_ind(trt_s, ctrl_s)
                    sig = bool(p_val < alpha)

                breached = sig and delta > 0
                status = "BREACHED" if breached else "OK"
                guard_rows.append({
                    "Metric": gm,
                    "Control": f"{ctrl_m:.4f}",
                    "Treatment": f"{trt_m:.4f}",
                    "Δ (abs)": f"{delta:+.4f}",
                    "Δ (rel %)": f"{rel:+.1f}%",
                    "p-value": f"{p_val:.4f}",
                    "Status": status,
                })

            if guard_rows:
                gdf = pd.DataFrame(guard_rows)

                def _color_status(val):
                    if "BREACHED" in str(val):
                        return "background-color: #7f1d1d; color: #fca5a5;"
                    if "OK" in str(val):
                        return "background-color: #064e3b; color: #6ee7b7;"
                    return ""

                st.dataframe(
                    gdf.style.map(_color_status, subset=["Status"]),
                    use_container_width=True, hide_index=True,
                )

    with bayes_tab:
        st.subheader("Bayesian Analysis")

        with st.expander("How the Bayesian model works"):
            st.markdown(
                """
**Beta-Binomial conjugate model**, exact with no MCMC required.

**Prior** Beta(α, β). Default α = β = 1 means uniform, expressing no prior knowledge.

**Update rule** After observing *x* successes in *n* users:
`Posterior ~ Beta(α + x, β + n − x)`

This is closed-form and posterior parameters update analytically.

**What we report**
- **P(Treatment > Control)** probability that treatment's true rate exceeds control's
- **Expected lift** posterior mean of (treatment rate minus control rate)
- **95% credible interval** unlike frequentist CIs, this *does* mean there is 95% posterior probability the true effect falls in this range

**When to use Bayesian vs frequentist**
Bayesian is more interpretable and handles small samples well because the prior regularizes.
Frequentist is the standard for regulatory and org-level decision rules.
                """
            )

        col_p, col_t = st.columns(2)
        with col_p:
            prior_a = st.number_input(
                "Prior α (pseudo-successes)",
                min_value=0.1, max_value=100.0, value=1.0, step=0.5,
                help="Prior Beta(α, β). α=1, β=1 means uniform with no prior knowledge. Increase α to encode a stronger prior toward conversion.",
            )
            prior_b = st.number_input(
                "Prior β (pseudo-failures)",
                min_value=0.1, max_value=100.0, value=1.0, step=0.5,
                help="Increase to encode a prior belief of low conversion.",
            )
        with col_t:
            threshold = st.slider(
                "Decision threshold P(B > A)",
                min_value=0.50, max_value=0.99, value=0.95, step=0.01,
                help=(
                    "Minimum posterior probability you require before shipping. "
                    "0.95 is a common default. Lower thresholds accept more uncertainty "
                    "while higher thresholds demand stronger evidence."
                ),
            )

        try:
            bayes = bayesian_binary_metric_readout(
                user_metrics=df, metric_col=primary_metric,
                variant_col=variant_col,
                control_variant=control_val, treatment_variant=treatment_val,
                prior_alpha=prior_a, prior_beta=prior_b,
            )

            post = bayes["posterior"]
            prob_better = bayes["prob_treatment_better"]
            exp_lift = bayes["expected_absolute_lift"]
            ci_lo, ci_hi = bayes["credible_interval_95_absolute_lift"]

            b1, b2, b3 = st.columns(3)
            b1.metric(
                "P(Treatment > Control)",
                f"{prob_better:.1%}",
                help="Posterior probability the treatment's true rate exceeds control's, given data and prior.",
            )
            b2.metric(
                "Expected absolute lift",
                lift(exp_lift),
                help="Posterior mean of (treatment rate − control rate). Expected improvement if you ship.",
            )
            b3.metric(
                "95% Credible interval",
                f"[{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp",
                help="There is 95% posterior probability the true absolute lift is in this range.",
            )

            rec = prob_better >= threshold
            rec_label = "SHIP" if rec else "HOLD"
            rec_kind = "pass" if rec else "warn"
            st.markdown(
                badge(
                    f"{rec_label}, P(B>A) = {prob_better:.1%} "
                    f"{'≥' if rec else '<'} threshold {threshold:.0%}",
                    rec_kind,
                ),
                unsafe_allow_html=True,
            )

            # Posterior distribution plot
            x = np.linspace(0, 1, 1000)
            ctrl_post_rv = scipy_beta(post["control_alpha"], post["control_beta"])
            trt_post_rv = scipy_beta(post["treatment_alpha"], post["treatment_beta"])

            x_lo = max(0.0, min(ctrl_post_rv.ppf(0.001), trt_post_rv.ppf(0.001)) - 0.02)
            x_hi = min(1.0, max(ctrl_post_rv.ppf(0.999), trt_post_rv.ppf(0.999)) + 0.02)
            x_zoom = np.linspace(x_lo, x_hi, 600)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_zoom, y=ctrl_post_rv.pdf(x_zoom),
                mode="lines", name=f"Control posterior",
                line=dict(color=COLOR_CONTROL, width=2),
                fill="tozeroy", fillcolor="rgba(148,163,184,0.15)",
            ))
            fig.add_trace(go.Scatter(
                x=x_zoom, y=trt_post_rv.pdf(x_zoom),
                mode="lines", name=f"Treatment posterior",
                line=dict(color=COLOR_TREATMENT, width=2),
                fill="tozeroy", fillcolor="rgba(129,140,248,0.15)",
            ))
            fig.update_layout(
                **chart_defaults(),
                xaxis_title="Conversion Rate", xaxis_tickformat=".1%",
                yaxis_title="Posterior Density",
                height=320, legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Beta posteriors for each variant. Overlap = uncertainty. "
                "Separation = evidence of true difference."
            )

            # Lift posterior histogram
            rng = np.random.default_rng(42)
            trt_draws = rng.beta(post["treatment_alpha"], post["treatment_beta"], 100_000)
            ctrl_draws = rng.beta(post["control_alpha"], post["control_beta"], 100_000)
            lift_draws = (trt_draws - ctrl_draws) * 100

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=lift_draws, nbinsx=80,
                marker_color=COLOR_TREATMENT, opacity=0.8,
                showlegend=False,
            ))
            fig2.add_vline(x=0, line_dash="dash", line_color=COLOR_NEUTRAL)
            fig2.add_vline(
                x=exp_lift * 100, line_dash="dot", line_color=COLOR_AMBER,
                annotation_text=f"E[lift] = {exp_lift*100:+.2f}pp",
                annotation_font_color=COLOR_AMBER,
            )
            fig2.update_layout(
                **chart_defaults(),
                xaxis_title="Absolute Lift (percentage points)",
                yaxis_title="Frequency",
                height=260, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Posterior lift distribution. Area right of zero = P(B>A) = {prob_better:.1%}."
            )

        except Exception as exc:
            st.error(f"Bayesian analysis failed: {exc}")

elif page == "Sequential Testing":
    st.title("Sequential Testing")
    st.caption("O'Brien-Fleming boundaries for valid early stopping without inflating your false positive rate.")

    with st.expander("How O'Brien-Fleming works"):
        st.markdown(
            r"""
**Sequential testing** lets you look at results multiple times without inflating the false positive rate.

**O'Brien-Fleming boundary** at information fraction *t* (fraction of planned sample collected)

$$\text{boundary}(t) = \frac{z_{1-\alpha/2}}{\sqrt{t}}$$

- At *t* = 0.25 (25% through), the boundary is roughly 2x stricter than the final threshold
- At *t* = 1.0 (all data in), the boundary equals the standard z-threshold

**Information fraction** = n_current / n_planned. You can stop early if |z| > boundary(t).
Keep collecting data otherwise until the final planned look.

This approach conserves your false-positive budget by spending less of it on early looks.
            """
        )

    col_s1, col_s2 = st.columns(2, gap="large")

    with col_s1:
        a_seq = st.select_slider(
            "α", options=[0.01, 0.05, 0.10], value=0.05, key="a_seq",
            help="Overall significance level for the sequential test.",
        )
        n_looks = st.number_input(
            "Number of planned looks",
            min_value=2, max_value=20, value=5, step=1,
            help="How many times you plan to check results, including the final look.",
        )

    with col_s2:
        st.markdown("**Evaluate a specific look**")
        info_frac = st.slider(
            "Information fraction (n_current / n_planned)",
            min_value=0.05, max_value=1.00, value=0.50, step=0.05,
            help="0.5 = halfway through the experiment. 1.0 = all data in.",
        )
        z_stat = st.number_input(
            "Observed z-statistic",
            min_value=-10.0, max_value=10.0, value=2.5, step=0.1,
            help="z-statistic from your current two-proportion test at this look.",
        )

        ev = evaluate_sequential_look(z_stat, info_frac, a_seq)
        boundary_val = ev["boundary"]
        can_stop = ev["can_stop_early"]

        m1, m2 = st.columns(2)
        m1.metric(
            "O'B-F boundary",
            f"±{boundary_val:.3f}",
            help="Must exceed this |z| to declare significance at this look.",
        )
        m2.metric("Your |z|", f"{abs(z_stat):.3f}")

        if can_stop:
            st.markdown(
                badge(f"CAN STOP — |z| = {abs(z_stat):.3f} exceeds {boundary_val:.3f}", "pass"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                badge(f"CONTINUE — |z| = {abs(z_stat):.3f} below {boundary_val:.3f}", "neutral"),
                unsafe_allow_html=True,
            )

    # Boundary curve
    fracs = np.linspace(1 / n_looks, 1.0, n_looks)
    bounds = [obrien_fleming_boundary(f, a_seq) for f in fracs]
    std_z = scipy_norm.ppf(1 - a_seq / 2)

    fig = go.Figure()
    fig.add_hrect(y0=-std_z, y1=std_z, fillcolor="rgba(100,116,139,0.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=fracs * 100, y=bounds,
        mode="lines+markers", line=dict(color=COLOR_TREATMENT, width=2.5),
        marker=dict(size=8), name="Upper boundary",
    ))
    fig.add_trace(go.Scatter(
        x=fracs * 100, y=[-b for b in bounds],
        mode="lines+markers", line=dict(color=COLOR_TREATMENT, width=2.5, dash="dot"),
        marker=dict(size=8), name="Lower boundary",
    ))
    fig.add_hline(y=std_z, line_dash="dash", line_color=COLOR_NEUTRAL,
                  annotation_text=f"Standard z = {std_z:.2f}")
    fig.add_hline(y=-std_z, line_dash="dash", line_color=COLOR_NEUTRAL)
    fig.add_trace(go.Scatter(
        x=[info_frac * 100], y=[z_stat],
        mode="markers",
        marker=dict(
            color=COLOR_POSITIVE if can_stop else COLOR_NEGATIVE,
            size=14, symbol="star",
        ),
        name=f"Your look (z = {z_stat:.2f})",
    ))
    fig.update_layout(
        **chart_defaults(),
        xaxis_title="Information Fraction (%)",
        yaxis_title="z-statistic",
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The boundary tightens early (when data is sparse) and relaxes to the standard z-threshold "
        "at the final look. Your z-statistic must cross the boundary to stop early."
    )

elif page == "Segment Analysis":
    st.title("Segment Analysis")
    st.caption("Detect heterogeneous treatment effects hidden by the overall average.")

    if not data_ready():
        st.warning("Load your data on the **Data and Setup** page first.")
        st.stop()

    df = st.session_state.data
    variant_col = st.session_state.variant_col
    control_val = st.session_state.control_val
    treatment_val = st.session_state.treatment_val
    primary_metric = st.session_state.primary_metric
    segment_cols = st.session_state.segment_cols

    with st.expander("Why segment analysis matters and its limits"):
        st.markdown(
            """
The **average treatment effect** can mask directionally opposite effects across subgroups.

**Classic pattern** A new onboarding flow:
- Helps low-intent users who need guidance at **+5pp**
- Hurts high-intent users who find it condescending at **-3pp**
- Average lands at **+1pp**, small and possibly insignificant

Segment analysis surfaces these heterogeneous effects so you can
- Ship only to benefiting segments
- Redesign to remove harms
- Generate hypotheses for follow-up experiments

**Caution on multiple comparisons** With many segments, some will appear significant
by chance due to the look-elsewhere effect. Treat segment findings as hypotheses for future
pre-registered experiments, not definitive conclusions, unless segments were
pre-specified in your experiment design.
            """
        )

    if not segment_cols:
        st.info(
            "No segment columns configured. "
            "Go to **Data and Setup** and add categorical columns under *Segment Columns*."
        )
        st.stop()

    selected = st.multiselect(
        "Segment columns to analyze",
        options=segment_cols,
        default=segment_cols[:3],
        help="Compute treatment effects separately for each value within these columns.",
    )

    if not selected:
        st.info("Select at least one segment column above.")
        st.stop()

    try:
        rows = analyze_segment_effects(
            user_metrics=df, metric_col=primary_metric,
            segment_cols=selected,
            variant_col=variant_col,
            control_variant=control_val, treatment_variant=treatment_val,
        )

        if not rows:
            st.warning("No segment data returned. Check column values.")
            st.stop()

        seg_df = pd.DataFrame(rows)

        for seg_col in selected:
            sub = seg_df[seg_df["segment_column"] == seg_col].copy()
            if sub.empty:
                continue

            st.subheader(f"By `{seg_col}`")
            sub = sub.sort_values("absolute_lift")

            colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in sub["absolute_lift"]]

            fig = go.Figure(go.Bar(
                x=sub["absolute_lift"] * 100,
                y=sub["segment_value"].astype(str),
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}pp" for v in sub["absolute_lift"] * 100],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Lift: %{x:+.2f}pp<br>"
                    "Control rate: " + sub["control_rate"].map("{:.2%}".format) + "<br>"
                    "<extra></extra>"
                ),
            ))
            fig.add_vline(x=0, line_color=COLOR_NEUTRAL, line_width=1)
            fig.update_layout(
                **chart_defaults(),
                xaxis_title="Absolute Lift (percentage points)",
                height=max(220, len(sub) * 52),
                margin=dict(l=10, r=70, t=30, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            disp = sub[
                ["segment_value", "n_control", "n_treatment",
                 "control_rate", "treatment_rate", "absolute_lift", "relative_lift_pct"]
            ].copy()
            disp.columns = [
                "Segment", "n Control", "n Treatment",
                "Control Rate", "Treatment Rate", "Abs Lift", "Rel Lift %",
            ]
            disp["Control Rate"] = disp["Control Rate"].map("{:.2%}".format)
            disp["Treatment Rate"] = disp["Treatment Rate"].map("{:.2%}".format)
            disp["Abs Lift"] = disp["Abs Lift"].map("{:+.2%}".format)
            disp["Rel Lift %"] = disp["Rel Lift %"].map("{:+.1f}%".format)

            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.divider()

    except Exception as exc:
        st.error(f"Segment analysis failed: {exc}")

elif page == "Decision Summary":
    st.title("Decision Summary")
    st.caption("Synthesized ship / hold / no-ship recommendation from all evidence.")

    if not data_ready():
        st.warning("Load your data on the **Data and Setup** page first.")
        st.stop()

    df = st.session_state.data
    variant_col = st.session_state.variant_col
    control_val = st.session_state.control_val
    treatment_val = st.session_state.treatment_val
    primary_metric = st.session_state.primary_metric
    guardrail_metrics = st.session_state.guardrail_metrics
    alpha = st.session_state.alpha

    with st.expander("Decision framework"):
        st.markdown(
            f"""
1. **Guardrails first.** If any guardrail is significantly degraded (p < {alpha}, Δ > 0), the decision is **No-Ship** regardless of the primary metric.
2. **Primary metric wins.** A significant improvement (p < {alpha}, Δ > 0) with no guardrail breach means **Ship**.
3. **Primary metric declines.** A significant worsening means **No-Ship**.
4. **Inconclusive.** If the primary is not significant, **Hold** and collect more data or re-evaluate the MDE.
5. **Bayesian corroboration.** P(B > A) ≥ 95% is reported as a secondary signal alongside the frequentist decision.
            """
        )

    try:
        freq = analyze_binary_metric(
            df, metric_col=primary_metric,
            variant_col=variant_col,
            control_variant=control_val, treatment_variant=treatment_val,
            alpha=alpha,
        )
    except Exception as exc:
        st.error(f"Primary metric analysis failed: {exc}")
        st.stop()

    try:
        bayes = bayesian_binary_metric_readout(
            df, metric_col=primary_metric,
            variant_col=variant_col,
            control_variant=control_val, treatment_variant=treatment_val,
        )
        prob_better = bayes["prob_treatment_better"]
    except Exception:
        bayes = None
        prob_better = None

    # Guardrail checks
    guardrail_breached = False
    guard_rows = []
    for gm in guardrail_metrics:
        if gm not in df.columns:
            continue
        is_bin = df[gm].dropna().isin([0, 1]).all()
        ctrl_s = df[df[variant_col] == control_val][gm].dropna()
        trt_s = df[df[variant_col] == treatment_val][gm].dropna()
        delta = trt_s.mean() - ctrl_s.mean()
        if is_bin:
            try:
                gr = analyze_binary_metric(
                    df, metric_col=gm, variant_col=variant_col,
                    control_variant=control_val, treatment_variant=treatment_val, alpha=alpha,
                )
                p_val = gr["p_value"]
                sig = gr["significant"]
            except Exception:
                p_val, sig = float("nan"), False
        else:
            _, p_val = ttest_ind(trt_s, ctrl_s)
            sig = bool(p_val < alpha)
        breached = sig and delta > 0
        if breached:
            guardrail_breached = True
        guard_rows.append({"metric": gm, "delta": delta, "p_value": p_val, "breached": breached})

    # Decision logic
    primary_sig = freq["significant"]
    primary_pos = freq["absolute_lift"] > 0

    if guardrail_breached:
        decision = "NO-SHIP"
        bg, border, tc = "#7f1d1d", "#DC2626", "#fca5a5"
        rationale = "One or more guardrail metrics are significantly degraded. Investigate before shipping."
    elif primary_sig and primary_pos:
        decision = "SHIP"
        bg, border, tc = "#064e3b", "#10B981", "#6ee7b7"
        rationale = f"Primary metric ({primary_metric}) improved significantly. No guardrails breached."
    elif primary_sig and not primary_pos:
        decision = "NO-SHIP"
        bg, border, tc = "#7f1d1d", "#DC2626", "#fca5a5"
        rationale = f"Primary metric ({primary_metric}) declined significantly."
    else:
        decision = "HOLD"
        bg, border, tc = "#78350f", "#F59E0B", "#fcd34d"
        rationale = "Primary metric did not reach statistical significance. Collect more data or re-evaluate the MDE."

    st.markdown(
        f"""
        <div class="decision-banner" style="background:{bg}; border: 2px solid {border};">
            <div style="font-size:2.4rem; font-weight:800; color:{tc}; letter-spacing:0.12em;">{decision}</div>
            <div style="color:{tc}; margin-top:8px; font-size:0.95rem;">{rationale}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evidence columns
    col_prim, col_guard = st.columns(2, gap="large")

    with col_prim:
        st.markdown("**Primary Metric**")
        ci_lo, ci_hi = freq["ci_95_absolute_lift"]
        st.metric(
            primary_metric,
            lift(freq["absolute_lift"]),
            delta=f"p = {freq['p_value']:.4f}  {'Significant' if primary_sig else 'Not significant'}",
            delta_color="normal" if primary_pos else "inverse",
        )
        st.caption(
            f"{(1-alpha):.0%} CI [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp  "
            f"Control {pct(freq['control_rate'])} to Treatment {pct(freq['treatment_rate'])}"
        )

        if bayes is not None:
            st.markdown("**Bayesian Corroboration**")
            signal = prob_better >= 0.95
            st.metric(
                "P(Treatment > Control)",
                f"{prob_better:.1%}",
                delta="≥ 95% threshold" if signal else "< 95% threshold",
                delta_color="normal" if signal else "off",
                help="Posterior probability the treatment's true rate exceeds control's.",
            )
            exp_lift_b = bayes["expected_absolute_lift"]
            ci_lo_b, ci_hi_b = bayes["credible_interval_95_absolute_lift"]
            st.caption(
                f"Expected lift {lift(exp_lift_b)}   "
                f"95% CrI [{ci_lo_b*100:+.2f}, {ci_hi_b*100:+.2f}] pp"
            )

    with col_guard:
        st.markdown("**Guardrail Status**")
        if not guard_rows:
            st.caption("No guardrail metrics configured.")
        else:
            for row in guard_rows:
                kind = "fail" if row["breached"] else "pass"
                status_txt = "BREACHED" if row["breached"] else "OK"
                st.markdown(
                    f"`{row['metric']}` &nbsp; Δ = {row['delta']:+.4f} &nbsp; "
                    f"p = {row['p_value']:.4f} &nbsp; "
                    + badge(status_txt, kind),
                    unsafe_allow_html=True,
                )

    # Experiment overview
    st.divider()
    st.subheader("Experiment Overview")

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Control n", f"{freq['n_control']:,}")
    ov2.metric("Treatment n", f"{freq['n_treatment']:,}")
    ov3.metric("Control rate", pct(freq["control_rate"]))
    ov4.metric("Treatment rate", pct(freq["treatment_rate"]))

    # Forest plot: primary + guardrails on one chart
    all_metrics_for_plot = [(primary_metric, freq["absolute_lift"], freq["ci_95_absolute_lift"], primary_sig)]
    for row in guard_rows:
        gm = row["metric"]
        if gm in df.columns:
            try:
                if df[gm].dropna().isin([0, 1]).all():
                    gr = analyze_binary_metric(
                        df, metric_col=gm, variant_col=variant_col,
                        control_variant=control_val, treatment_variant=treatment_val,
                        alpha=alpha,
                    )
                    all_metrics_for_plot.append((gm, gr["absolute_lift"], gr["ci_95_absolute_lift"], gr["significant"]))
            except Exception:
                pass

    if len(all_metrics_for_plot) > 1:
        st.divider()
        st.subheader("Forest Plot (All Binary Metrics)")
        labels = [m[0] for m in all_metrics_for_plot]
        centers = [m[1] * 100 for m in all_metrics_for_plot]
        errors_hi = [(m[2][1] - m[1]) * 100 for m in all_metrics_for_plot]
        errors_lo = [(m[1] - m[2][0]) * 100 for m in all_metrics_for_plot]
        dot_colors = [
            (COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE) if sig else COLOR_NEUTRAL
            for c, (_, _, _, sig) in zip(centers, all_metrics_for_plot)
        ]

        fig = go.Figure()
        fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(labels) - 0.5,
                      line=dict(color=COLOR_NEUTRAL, width=1, dash="dash"))
        for i, (label, center, eh, el, color) in enumerate(
            zip(labels, centers, errors_hi, errors_lo, dot_colors)
        ):
            fig.add_trace(go.Scatter(
                x=[center], y=[label],
                mode="markers",
                marker=dict(color=color, size=12, symbol="diamond"),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[eh], arrayminus=[el],
                    color=COLOR_CONTROL, thickness=2, width=8,
                ),
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Lift: {center:+.2f}pp<extra></extra>",
            ))
        fig.update_layout(
            **chart_defaults(),
            xaxis_title="Absolute Lift (percentage points)",
            yaxis=dict(autorange="reversed"),
            height=max(200, len(labels) * 70),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green = significant positive. Red = significant negative. "
            "Gray = not significant. Whiskers = 95% CI."
        )
