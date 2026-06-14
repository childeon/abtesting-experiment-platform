import numpy as np


def bayesian_binary_metric_readout(
    user_metrics,
    metric_col,
    variant_col="variant",
    control_variant="control",
    treatment_variant="treatment",
    prior_alpha=1,
    prior_beta=1,
    n_draws=100_000,
    seed=42,
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

    control_posterior_alpha = prior_alpha + x_control
    control_posterior_beta = prior_beta + n_control - x_control
    treatment_posterior_alpha = prior_alpha + x_treatment
    treatment_posterior_beta = prior_beta + n_treatment - x_treatment

    rng = np.random.default_rng(seed)
    control_draws = rng.beta(
        control_posterior_alpha,
        control_posterior_beta,
        size=n_draws,
    )
    treatment_draws = rng.beta(
        treatment_posterior_alpha,
        treatment_posterior_beta,
        size=n_draws,
    )
    lift_draws = treatment_draws - control_draws

    return {
        "metric": metric_col,
        "control_variant": control_variant,
        "treatment_variant": treatment_variant,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "x_control": x_control,
        "x_treatment": x_treatment,
        "prior": {
            "alpha": prior_alpha,
            "beta": prior_beta,
        },
        "posterior": {
            "control_alpha": control_posterior_alpha,
            "control_beta": control_posterior_beta,
            "treatment_alpha": treatment_posterior_alpha,
            "treatment_beta": treatment_posterior_beta,
        },
        "prob_treatment_better": float((lift_draws > 0).mean()),
        "expected_absolute_lift": float(lift_draws.mean()),
        "credible_interval_95_absolute_lift": (
            float(np.quantile(lift_draws, 0.025)),
            float(np.quantile(lift_draws, 0.975)),
        ),
        "n_draws": n_draws,
    }
