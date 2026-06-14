import pandas as pd


FIRST_7_DAYS_HOURS = 168
DAY_7_START_HOURS = 168
DAY_7_END_HOURS = 192
RETENTION_EVENTS = {"started_lesson", "completed_lesson"}


def _validate_required_columns(dataframe, required_columns, dataframe_name):
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing_columns}"
        )


def build_user_metrics(users, assignments, events):
    _validate_required_columns(
        users,
        ["user_id", "signup_timestamp", "country", "platform", "user_segment"],
        "users",
    )
    _validate_required_columns(
        assignments,
        ["user_id", "experiment_id", "variant", "assigned_at"],
        "assignments",
    )
    _validate_required_columns(
        events,
        ["user_id", "event_name", "event_timestamp", "load_time_ms", "amount_usd"],
        "events",
    )
    events = events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])
    events["load_time_ms"] = pd.to_numeric(events["load_time_ms"], errors="coerce")
    events["amount_usd"] = pd.to_numeric(events["amount_usd"], errors="coerce")

    assigned_users = assignments.merge(
        users[
            [
                "user_id",
                "signup_timestamp",
                "country",
                "platform",
                "user_segment",
            ]
        ],
        on="user_id",
        how="left",
    )

    events_with_context = events.merge(
        assigned_users[["user_id", "signup_timestamp"]],
        on="user_id",
        how="inner",
    )
    events_with_context["hours_after_signup"] = (
        events_with_context["event_timestamp"]
        - events_with_context["signup_timestamp"]
    ).dt.total_seconds() / 3600

    events_first_7_days = events_with_context[
        (events_with_context["hours_after_signup"] >= 0)
        & (events_with_context["hours_after_signup"] < FIRST_7_DAYS_HOURS)
    ]
    events_day_7 = events_with_context[
        (events_with_context["hours_after_signup"] >= DAY_7_START_HOURS)
        & (events_with_context["hours_after_signup"] < DAY_7_END_HOURS)
    ]

    day_7_retained_users = set(
        events_day_7[
            events_day_7["event_name"].isin(RETENTION_EVENTS)
        ]["user_id"].unique()
    )

    sessions = (
        events_first_7_days[events_first_7_days["event_name"] == "session_start"]
        .groupby("user_id")
        .size()
        .rename("sessions_first_7_days")
    )
    crashes = (
        events_first_7_days[events_first_7_days["event_name"] == "app_crash"]
        .groupby("user_id")
        .size()
        .rename("crashes_first_7_days")
    )
    revenue = (
        events_first_7_days[
            events_first_7_days["event_name"] == "subscription_started"
        ]
        .groupby("user_id")["amount_usd"]
        .sum()
        .rename("revenue_first_7_days")
    )
    p95_load_time = (
        events_first_7_days[events_first_7_days["event_name"] == "page_load"]
        .groupby("user_id")["load_time_ms"]
        .quantile(0.95)
        .rename("load_time_p95_first_7_days")
    )

    user_metrics = assigned_users.copy()
    user_metrics["day_7_retained"] = user_metrics["user_id"].isin(
        day_7_retained_users
    ).astype(int)

    user_metrics = user_metrics.merge(sessions, on="user_id", how="left")
    user_metrics = user_metrics.merge(crashes, on="user_id", how="left")
    user_metrics = user_metrics.merge(revenue, on="user_id", how="left")
    user_metrics = user_metrics.merge(p95_load_time, on="user_id", how="left")

    fill_zero_columns = [
        "sessions_first_7_days",
        "crashes_first_7_days",
        "revenue_first_7_days",
    ]
    user_metrics[fill_zero_columns] = user_metrics[fill_zero_columns].fillna(0)

    return user_metrics


def aggregate_experiment_metrics(user_metrics):
    grouped = user_metrics.groupby(["experiment_id", "variant"])

    summary = grouped.agg(
        users=("user_id", "count"),
        day_7_retention_rate=("day_7_retained", "mean"),
        sessions_per_user=("sessions_first_7_days", "mean"),
        crashes=("crashes_first_7_days", "sum"),
        sessions=("sessions_first_7_days", "sum"),
        revenue_per_user=("revenue_first_7_days", "mean"),
        p95_load_time_ms=("load_time_p95_first_7_days", "quantile"),
    ).reset_index()

    summary["crash_rate"] = summary["crashes"] / summary["sessions"].where(
        summary["sessions"] > 0
    )

    return summary[
        [
            "experiment_id",
            "variant",
            "users",
            "day_7_retention_rate",
            "sessions_per_user",
            "crash_rate",
            "revenue_per_user",
            "p95_load_time_ms",
        ]
    ]
