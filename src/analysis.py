import math

from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.stats.proportion import proportions_ztest


def analyze_binary_metric(
    user_metrics,
    metric_col,
    variant_col="variant",
    control_variant="control",
    treatment_variant="treatment",
    alpha=0.05,
):
    required_columns = {variant_col, metric_col}
    missing_columns = required_columns - set(user_metrics.columns)
    if missing_columns:
        raise ValueError(f"user_metrics is missing columns: {missing_columns}")

    control = user_metrics[user_metrics[variant_col] == control_variant][metric_col]
    treatment = user_metrics[user_metrics[variant_col] == treatment_variant][metric_col]

    if control.empty:
        raise ValueError(f"No rows found for control variant: {control_variant}")

    if treatment.empty:
        raise ValueError(f"No rows found for treatment variant: {treatment_variant}")

    n_control = len(control)
    n_treatment = len(treatment)
    x_control = int(control.sum())
    x_treatment = int(treatment.sum())

    control_rate = x_control / n_control
    treatment_rate = x_treatment / n_treatment
    absolute_lift = treatment_rate - control_rate
    relative_lift_pct = (
        absolute_lift / control_rate * 100 if control_rate != 0 else math.nan
    )

    z_statistic, p_value = proportions_ztest(
        count=[x_treatment, x_control],
        nobs=[n_treatment, n_control],
        alternative="two-sided",
    )
    ci_lower, ci_upper = confint_proportions_2indep(
        count1=x_treatment,
        nobs1=n_treatment,
        count2=x_control,
        nobs2=n_control,
        compare="diff",
        alpha=alpha,
        method="wald",
    )

    return {
        "metric": metric_col,
        "control_variant": control_variant,
        "treatment_variant": treatment_variant,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "x_control": x_control,
        "x_treatment": x_treatment,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_lift": absolute_lift,
        "relative_lift_pct": relative_lift_pct,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "ci_95_absolute_lift": (ci_lower, ci_upper),
    }
