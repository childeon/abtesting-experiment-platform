from datetime import datetime


def _format_pct(value):
    return f"{value * 100:.2f}%"


def _format_pp(value):
    return f"{value * 100:+.2f}pp"


def _format_money(value):
    return f"${value:.2f}"


def generate_decision_report(
    experiment_config,
    frequentist_result,
    bayesian_result,
    srm_result,
    experiment_metrics,
    segment_effects,
):
    guardrail_lines = []
    control_row = experiment_metrics[
        experiment_metrics["variant"] == "control"
    ].iloc[0]
    treatment_row = experiment_metrics[
        experiment_metrics["variant"] == "treatment"
    ].iloc[0]

    guardrail_lines.append(
        "- Revenue/user: "
        f"{_format_money(control_row['revenue_per_user'])} control vs "
        f"{_format_money(treatment_row['revenue_per_user'])} treatment"
    )
    guardrail_lines.append(
        "- Crash rate: "
        f"{_format_pct(control_row['crash_rate'])} control vs "
        f"{_format_pct(treatment_row['crash_rate'])} treatment"
    )
    guardrail_lines.append(
        "- P95 load time: "
        f"{control_row['p95_load_time_ms']:.1f} ms control vs "
        f"{treatment_row['p95_load_time_ms']:.1f} ms treatment"
    )

    sorted_segments = sorted(
        segment_effects,
        key=lambda row: row["absolute_lift"],
    )
    weakest_segment = sorted_segments[0]
    strongest_segment = sorted_segments[-1]

    if (
        frequentist_result["significant"]
        and frequentist_result["absolute_lift"] > 0
        and not srm_result["srm_detected"]
    ):
        decision = "SHIP WITH MONITORING"
        recommendation = (
            "Ship the onboarding checklist. Continue monitoring guardrails and "
            "watch the weakest segment after rollout."
        )
    elif srm_result["srm_detected"]:
        decision = "DO NOT SHIP"
        recommendation = (
            "Do not make a launch decision until the sample ratio mismatch is "
            "investigated."
        )
    else:
        decision = "INCONCLUSIVE"
        recommendation = (
            "Do not ship yet. Continue the experiment or revisit the design "
            "depending on runtime and power."
        )

    report = f"""# Experiment Decision Report: {experiment_config["name"]}

Generated: {datetime.now().strftime("%Y-%m-%d")}

## Decision

**{decision}**

{recommendation}

## Experiment

- Experiment ID: `{experiment_config["experiment_id"]}`
- Hypothesis: {experiment_config["hypothesis"]}
- Population: {experiment_config["population"]}
- Primary metric: `{experiment_config["primary_metric"]}`

## Data Quality

- Sample ratio mismatch detected: {srm_result["srm_detected"]}
- SRM p-value: {srm_result["p_value"]:.4g}

## Primary Result

![Day-7 retention by variant](../figures/primary_metric_retention.png)

- Control: {frequentist_result["x_control"]:,} / {frequentist_result["n_control"]:,} retained ({_format_pct(frequentist_result["control_rate"])})
- Treatment: {frequentist_result["x_treatment"]:,} / {frequentist_result["n_treatment"]:,} retained ({_format_pct(frequentist_result["treatment_rate"])})
- Absolute lift: {_format_pp(frequentist_result["absolute_lift"])}
- Relative lift: {frequentist_result["relative_lift_pct"]:+.2f}%
- p-value: {frequentist_result["p_value"]:.4f}
- 95% CI for absolute lift: {_format_pp(frequentist_result["ci_95_absolute_lift"][0])} to {_format_pp(frequentist_result["ci_95_absolute_lift"][1])}

## Bayesian Readout

- Probability treatment is better: {_format_pct(bayesian_result["prob_treatment_better"])}
- Expected absolute lift: {_format_pp(bayesian_result["expected_absolute_lift"])}
- 95% credible interval: {_format_pp(bayesian_result["credible_interval_95_absolute_lift"][0])} to {_format_pp(bayesian_result["credible_interval_95_absolute_lift"][1])}

## Guardrails

![Guardrail metrics by variant](../figures/guardrail_metrics.png)

{chr(10).join(guardrail_lines)}

## Segment Notes

![Segment lift chart](../figures/segment_lift.png)

- Strongest segment: `{strongest_segment["segment_column"]} = {strongest_segment["segment_value"]}` ({_format_pp(strongest_segment["absolute_lift"])})
- Weakest segment: `{weakest_segment["segment_column"]} = {weakest_segment["segment_value"]}` ({_format_pp(weakest_segment["absolute_lift"])})

## Recommendation

Ship the treatment, but monitor the weakest segment after rollout and confirm guardrails remain healthy at larger scale.
"""
    return report
