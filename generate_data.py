from pathlib import Path

from src.assignment import assign_users
from src.config import EXPERIMENT_CONFIG
from src.events import simulate_events
from src.metrics import aggregate_experiment_metrics, build_user_metrics
from src.simulate_users import simulate_users


DATA_DIR = Path(__file__).parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    users = simulate_users(
        n_users=10_000,
        start_date="2026-01-01",
        seed=42,
    )
    assignments = assign_users(
        users=users,
        experiment_id=EXPERIMENT_CONFIG["experiment_id"],
        traffic_allocation=EXPERIMENT_CONFIG["traffic_allocation"],
    )
    events = simulate_events(
        users=users,
        assignments=assignments,
        seed=42,
        baseline_day_7_retention_rate=0.12,
        treatment_day_7_retention_lift=0.02,
    )
    user_metrics = build_user_metrics(
        users=users,
        assignments=assignments,
        events=events,
    )
    experiment_metrics = aggregate_experiment_metrics(user_metrics)

    users.to_csv(DATA_DIR / "users.csv", index=False)
    assignments.to_csv(DATA_DIR / "assignments.csv", index=False)
    events.to_csv(DATA_DIR / "events.csv", index=False)
    user_metrics.to_csv(DATA_DIR / "user_metrics.csv", index=False)
    experiment_metrics.to_csv(DATA_DIR / "experiment_metrics.csv", index=False)

    print(f"Wrote data files to {DATA_DIR}")
    print(experiment_metrics)


if __name__ == "__main__":
    main()
