EXPERIMENT_CONFIG = {
    "experiment_id": "exp_onboarding_checklist_001",
    "name": "New User Onboarding Checklist",
    "hypothesis": (
        "The new user onboarding checklist will increase day-7 retention "
        "because it helps users discover the product's value faster."
    ),
    "population": "New users who signed up during the experiment",
    "variants": {
        "control": "Existing onboarding",
        "treatment": "Onboarding checklist",
    },
    "traffic_allocation": {
        "control": 0.5,
        "treatment": 0.5,
    },
    "primary_metric": "day_7_retention_rate",
    "metric_definitions": {
        "day_7_retention_rate": {
            "description": (
                "Share of new users with at least one qualifying learning "
                "activity in their day-7 window."
            ),
            "unit_of_analysis": "user",
            "denominator": (
                "New users who signed up during the experiment and were "
                "assigned to a variant."
            ),
            "numerator": (
                "Users in the denominator with at least one qualifying event "
                "between 168 and 192 hours after signup."
            ),
            "qualifying_events": [
                "started_lesson",
                "completed_lesson",
            ],
            "window_start_hours_after_signup": 168,
            "window_end_hours_after_signup": 192,
        },
    },
    "secondary_metrics": [
        "day_1_retention_rate",
        "sessions_per_user",
    ],
    "guardrail_metrics": [
        "crash_rate",
        "p95_load_time_ms",
        "revenue_per_user",
    ],
    "decision_rule": (
        "Ship if day-7 retention improves and guardrail metrics do not degrade."
    ),
}
