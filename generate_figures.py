from pathlib import Path

import pandas as pd

from src.analysis import analyze_binary_metric
from src.monitoring import simulate_peeking_false_positive_rate
from src.segments import analyze_segment_effects
from src.visualizations import (
    plot_guardrails,
    plot_peeking_false_positive_rates,
    plot_primary_metric,
    plot_segment_lift,
)


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"


def main():
    FIGURE_DIR.mkdir(exist_ok=True)

    user_metrics = pd.read_csv(DATA_DIR / "user_metrics.csv")
    experiment_metrics = pd.read_csv(DATA_DIR / "experiment_metrics.csv")

    frequentist_result = analyze_binary_metric(user_metrics, "day_7_retained")
    segment_effects = analyze_segment_effects(
        user_metrics,
        "day_7_retained",
        ["platform", "country", "user_segment"],
    )
    peeking_result = simulate_peeking_false_positive_rate(
        n_days=14,
        users_per_day_per_variant=500,
        baseline_rate=0.12,
        alpha=0.05,
        n_simulations=1000,
        seed=42,
    )

    plot_primary_metric(
        frequentist_result,
        FIGURE_DIR / "primary_metric_retention.png",
    )
    plot_guardrails(
        experiment_metrics,
        FIGURE_DIR / "guardrail_metrics.png",
    )
    plot_segment_lift(
        segment_effects,
        FIGURE_DIR / "segment_lift.png",
    )
    plot_peeking_false_positive_rates(
        peeking_result,
        FIGURE_DIR / "peeking_false_positive_rates.png",
    )

    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
