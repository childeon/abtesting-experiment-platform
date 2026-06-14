import math

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def validate_sample_size_inputs(
    baseline_rate,
    minimum_detectable_effect,
    alpha,
    power,
    daily_eligible_users,
    metric_maturity_days,
    n_variants,
):
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")

    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive.")

    if baseline_rate + minimum_detectable_effect >= 1:
        raise ValueError("baseline_rate + minimum_detectable_effect must be below 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1.")

    if daily_eligible_users <= 0:
        raise ValueError("daily_eligible_users must be positive.")

    if metric_maturity_days < 0:
        raise ValueError("metric_maturity_days must be zero or positive.")

    if n_variants < 2:
        raise ValueError("n_variants must be at least 2.")


def calculate_sample_size(
    baseline_rate,
    minimum_detectable_effect,
    alpha,
    power,
    daily_eligible_users,
    metric_maturity_days,
    n_variants=2,
):
    validate_sample_size_inputs(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
        daily_eligible_users=daily_eligible_users,
        metric_maturity_days=metric_maturity_days,
        n_variants=n_variants,
    )

    target_rate = baseline_rate + minimum_detectable_effect
    # Same absolute lift can have different variance at different baselines.
    effect_size = proportion_effectsize(target_rate, baseline_rate)
    corrected_alpha = alpha / (n_variants - 1)

    analysis = NormalIndPower()
    n_per_variant = math.ceil(
        analysis.solve_power(
            effect_size=effect_size,
            alpha=corrected_alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
    )

    total_n = n_per_variant * n_variants
    enrollment_days = math.ceil(total_n / daily_eligible_users)
    total_calendar_days = enrollment_days + metric_maturity_days

    return {
        "baseline_rate": baseline_rate,
        "target_rate": target_rate,
        "minimum_detectable_effect": minimum_detectable_effect,
        "alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "power": power,
        "n_variants": n_variants,
        "n_per_variant": n_per_variant,
        "total_n": total_n,
        "daily_eligible_users": daily_eligible_users,
        "metric_maturity_days": metric_maturity_days,
        "enrollment_days": enrollment_days,
        "total_calendar_days": total_calendar_days,
    }
