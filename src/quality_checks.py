import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chisquare


def check_sample_ratio_mismatch(
    assignments,
    traffic_allocation,
    alpha=0.001,
):
    observed_counts = (
        assignments["variant"]
        .value_counts()
        .reindex(traffic_allocation.keys(), fill_value=0)
    )
    total_users = observed_counts.sum()
    expected_counts = {
        variant: total_users * allocation
        for variant, allocation in traffic_allocation.items()
    }

    chi_square_statistic, p_value = chisquare(
        f_obs=observed_counts.values,
        f_exp=list(expected_counts.values()),
    )

    return {
        "observed_counts": observed_counts.to_dict(),
        "expected_counts": expected_counts,
        "chi_square_statistic": chi_square_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "srm_detected": p_value < alpha,
    }


def run_aa_test(
    n_users,
    baseline_rate,
    alpha=0.05,
    n_simulations=1000,
    seed=42,
):
    if n_users <= 0:
        raise ValueError("n_users must be positive.")

    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    rng = np.random.default_rng(seed)
    false_positives = 0

    n_control = n_users // 2
    n_treatment = n_users - n_control

    for _ in range(n_simulations):
        control_retained = rng.binomial(n_control, baseline_rate)
        treatment_retained = rng.binomial(n_treatment, baseline_rate)

        _, p_value = proportions_ztest(
            count=[treatment_retained, control_retained],
            nobs=[n_treatment, n_control],
            alternative="two-sided",
        )

        if p_value < alpha:
            false_positives += 1

    false_positive_rate = false_positives / n_simulations

    return {
        "false_positive_rate": false_positive_rate,
        "expected_alpha": alpha,
        "n_simulations": n_simulations,
        "n_users": n_users,
        "baseline_rate": baseline_rate,

    }
