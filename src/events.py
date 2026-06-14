import numpy as np
import pandas as pd


SEGMENT_ACTIVITY_MULTIPLIER = {
    "low_intent": 0.75,
    "medium_intent": 1.0,
    "high_intent": 1.35,
}


def _append_event(
    events,
    user_id,
    event_name,
    event_timestamp,
    load_time_ms=None,
    amount_usd=None,
):
    events.append(
        {
            "user_id": user_id,
            "event_name": event_name,
            "event_timestamp": event_timestamp,
            "load_time_ms": load_time_ms,
            "amount_usd": amount_usd,
        }
    )


def simulate_events(
    users,
    assignments=None,
    seed=42,
    baseline_day_7_retention_rate=0.12,
    treatment_day_7_retention_lift=0.0,
    subscription_rate=0.06,
    crash_rate_per_session=0.015,
):
    required_columns = {"user_id", "signup_timestamp", "user_segment"}
    missing_columns = required_columns - set(users.columns)
    if missing_columns:
        raise ValueError(f"users is missing required columns: {missing_columns}")

    rng = np.random.default_rng(seed)
    events = []
    variant_by_user = {}
    if assignments is not None:
        required_assignment_columns = {"user_id", "variant"}
        missing_assignment_columns = required_assignment_columns - set(
            assignments.columns
        )
        if missing_assignment_columns:
            raise ValueError(
                "assignments is missing required columns: "
                f"{missing_assignment_columns}"
            )
        variant_by_user = assignments.set_index("user_id")["variant"].to_dict()

    for row in users.itertuples(index=False):
        user_id = row.user_id
        signup_timestamp = row.signup_timestamp
        segment = row.user_segment
        activity_multiplier = SEGMENT_ACTIVITY_MULTIPLIER[segment]
        variant = variant_by_user.get(user_id)

        n_sessions = rng.poisson(2.2 * activity_multiplier)

        for _ in range(n_sessions):
            hours_after_signup = rng.uniform(0, 7 * 24)
            session_timestamp = signup_timestamp + pd.Timedelta(
                hours=hours_after_signup
            )
            _append_event(events, user_id, "session_start", session_timestamp)

            load_time_ms = max(100, rng.normal(900, 250))
            _append_event(
                events,
                user_id,
                "page_load",
                session_timestamp + pd.Timedelta(seconds=1),
                load_time_ms=round(load_time_ms, 1),
            )

            if rng.random() < crash_rate_per_session:
                _append_event(
                    events,
                    user_id,
                    "app_crash",
                    session_timestamp + pd.Timedelta(minutes=rng.uniform(1, 8)),
                )

            if rng.random() < 0.45 * activity_multiplier:
                started_at = session_timestamp + pd.Timedelta(
                    minutes=rng.uniform(1, 5)
                )
                _append_event(events, user_id, "started_lesson", started_at)

                if rng.random() < 0.65:
                    _append_event(
                        events,
                        user_id,
                        "completed_lesson",
                        started_at + pd.Timedelta(minutes=rng.uniform(5, 20)),
                    )

        day_7_retention_rate = baseline_day_7_retention_rate
        if variant == "treatment":
            day_7_retention_rate += treatment_day_7_retention_lift
        day_7_retention_rate = min(day_7_retention_rate * activity_multiplier, 0.95)

        day_7_retained = rng.random() < day_7_retention_rate
        if day_7_retained:
            day_7_timestamp = signup_timestamp + pd.Timedelta(
                hours=rng.uniform(168, 192)
            )
            _append_event(events, user_id, "started_lesson", day_7_timestamp)

            if rng.random() < 0.70:
                _append_event(
                    events,
                    user_id,
                    "completed_lesson",
                    day_7_timestamp + pd.Timedelta(minutes=rng.uniform(5, 20)),
                )

        subscribed = rng.random() < (subscription_rate * activity_multiplier)
        if subscribed:
            subscription_timestamp = signup_timestamp + pd.Timedelta(
                hours=rng.uniform(0, 7 * 24)
            )
            _append_event(
                events,
                user_id,
                "subscription_started",
                subscription_timestamp,
                amount_usd=10.00,
            )

    return pd.DataFrame(events)
