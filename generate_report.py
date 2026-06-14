from pathlib import Path

import pandas as pd

from src.analysis import analyze_binary_metric
from src.bayesian import bayesian_binary_metric_readout
from src.config import EXPERIMENT_CONFIG
from src.quality_checks import check_sample_ratio_mismatch
from src.report import generate_decision_report
from src.segments import analyze_segment_effects


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    assignments = pd.read_csv(DATA_DIR / "assignments.csv")
    user_metrics = pd.read_csv(DATA_DIR / "user_metrics.csv")
    experiment_metrics = pd.read_csv(DATA_DIR / "experiment_metrics.csv")

    frequentist_result = analyze_binary_metric(user_metrics, "day_7_retained")
    bayesian_result = bayesian_binary_metric_readout(user_metrics, "day_7_retained")
    srm_result = check_sample_ratio_mismatch(
        assignments,
        EXPERIMENT_CONFIG["traffic_allocation"],
    )
    segment_effects = analyze_segment_effects(
        user_metrics,
        "day_7_retained",
        ["platform", "country", "user_segment"],
    )

    report = generate_decision_report(
        experiment_config=EXPERIMENT_CONFIG,
        frequentist_result=frequentist_result,
        bayesian_result=bayesian_result,
        srm_result=srm_result,
        experiment_metrics=experiment_metrics,
        segment_effects=segment_effects,
    )

    report_path = REPORT_DIR / "experiment_decision_report.md"
    report_path.write_text(report)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
