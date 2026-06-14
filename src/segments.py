import math


def analyze_segment_effects(
    user_metrics,
    metric_col,
    segment_cols,
    variant_col="variant",
    control_variant="control",
    treatment_variant="treatment",
):
    required_columns = {metric_col, variant_col, *segment_cols}
    missing_columns = required_columns - set(user_metrics.columns)
    if missing_columns:
        raise ValueError(f"user_metrics is missing columns: {missing_columns}")

    rows = []

    for segment_col in segment_cols:
        for segment_value, segment_data in user_metrics.groupby(segment_col):
            control = segment_data[
                segment_data[variant_col] == control_variant
            ][metric_col]
            treatment = segment_data[
                segment_data[variant_col] == treatment_variant
            ][metric_col]

            if control.empty or treatment.empty:
                continue

            control_rate = control.mean()
            treatment_rate = treatment.mean()
            absolute_lift = treatment_rate - control_rate
            relative_lift_pct = (
                absolute_lift / control_rate * 100
                if control_rate != 0
                else math.nan
            )

            rows.append(
                {
                    "segment_column": segment_col,
                    "segment_value": segment_value,
                    "n_control": len(control),
                    "n_treatment": len(treatment),
                    "control_rate": control_rate,
                    "treatment_rate": treatment_rate,
                    "absolute_lift": absolute_lift,
                    "relative_lift_pct": relative_lift_pct,
                }
            )

    return rows
