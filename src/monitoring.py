import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest


def _two_proportion_p_value(
    control_successes,
    control_n,
    treatment_successes,
    treatment_n,
):
    _, p_value = proportions_ztest(
        count=[treatment_successes, control_successes],
        nobs=[treatment_n, control_n],
        alternative="two-sided",
    )
    return p_value


def simulate_peeking_false_positive_rate(
    n_days,
    users_per_day_per_variant,
    baseline_rate,
    alpha=0.05,
    n_simulations=1000,
    seed=42,
):
    if n_days <= 0:
        raise ValueError("n_days must be positive.")

    if users_per_day_per_variant <= 0:
        raise ValueError("users_per_day_per_variant must be positive.")

    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    rng = np.random.default_rng(seed)
    final_only_false_positives = 0
    peeking_false_positives = 0

    for _ in range(n_simulations):
        control_daily = rng.binomial(
            n=users_per_day_per_variant,
            p=baseline_rate,
            size=n_days,
        )
        treatment_daily = rng.binomial(
            n=users_per_day_per_variant,
            p=baseline_rate,
            size=n_days,
        )

        crossed_during_peeking = False

        for day_index in range(n_days):
            days_observed = day_index + 1
            control_successes = control_daily[:days_observed].sum()
            treatment_successes = treatment_daily[:days_observed].sum()
            control_n = users_per_day_per_variant * days_observed
            treatment_n = users_per_day_per_variant * days_observed

            p_value = _two_proportion_p_value(
                control_successes=control_successes,
                control_n=control_n,
                treatment_successes=treatment_successes,
                treatment_n=treatment_n,
            )

            if p_value < alpha:
                crossed_during_peeking = True

        final_p_value = _two_proportion_p_value(
            control_successes=control_daily.sum(),
            control_n=users_per_day_per_variant * n_days,
            treatment_successes=treatment_daily.sum(),
            treatment_n=users_per_day_per_variant * n_days,
        )

        if final_p_value < alpha:
            final_only_false_positives += 1

        if crossed_during_peeking:
            peeking_false_positives += 1

    false_positive_rate_final_only = final_only_false_positives / n_simulations
    false_positive_rate_with_peeking = peeking_false_positives / n_simulations
    inflation_factor = (
        false_positive_rate_with_peeking / false_positive_rate_final_only
        if false_positive_rate_final_only > 0
        else np.inf
    )

    return {
        "false_positive_rate_final_only": false_positive_rate_final_only,
        "false_positive_rate_with_peeking": false_positive_rate_with_peeking,
        "inflation_factor": inflation_factor,
        "alpha": alpha,
        "n_simulations": n_simulations,
        "n_days": n_days,
        "users_per_day_per_variant": users_per_day_per_variant,
        "baseline_rate": baseline_rate,
    }


def obrien_fleming_boundary(information_fraction, alpha=0.05):
    if not 0 < information_fraction <= 1:
        raise ValueError("information_fraction must be greater than 0 and at most 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    final_boundary = norm.ppf(1 - alpha / 2)
    return final_boundary / np.sqrt(information_fraction)


def evaluate_sequential_look(z_statistic, information_fraction, alpha=0.05):
    boundary = obrien_fleming_boundary(
        information_fraction=information_fraction,
        alpha=alpha,
    )

    return {
        "z_statistic": z_statistic,
        "information_fraction": information_fraction,
        "boundary": boundary,
        "alpha": alpha,
        "can_stop_early": abs(z_statistic) > boundary,
    }
