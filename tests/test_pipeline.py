import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.assignment import assign_users, assign_variant, summarize_assignment_balance
from src.metrics import aggregate_experiment_metrics, build_user_metrics
from src.monitoring import evaluate_sequential_look
from src.quality_checks import check_sample_ratio_mismatch
from src.simulate_users import simulate_users


class ExperimentPipelineTests(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        allocation = {"control": 0.5, "treatment": 0.5}

        first_assignment = assign_variant(
            user_id=123,
            experiment_id="exp_test",
            traffic_allocation=allocation,
        )
        second_assignment = assign_variant(
            user_id=123,
            experiment_id="exp_test",
            traffic_allocation=allocation,
        )

        self.assertEqual(first_assignment, second_assignment)

    def test_assignment_balance_is_close_to_expected(self):
        users = simulate_users(10_000, "2026-01-01")
        assignments = assign_users(
            users=users,
            experiment_id="exp_test",
            traffic_allocation={"control": 0.5, "treatment": 0.5},
        )
        balance = summarize_assignment_balance(assignments).set_index("variant")

        self.assertLess(abs(balance.loc["control", "assignment_share"] - 0.5), 0.02)
        self.assertLess(
            abs(balance.loc["treatment", "assignment_share"] - 0.5), 0.02
        )

    def test_srm_check_flags_broken_split(self):
        assignments = pd.DataFrame(
            {"variant": ["control"] * 6200 + ["treatment"] * 3800}
        )
        result = check_sample_ratio_mismatch(
            assignments=assignments,
            traffic_allocation={"control": 0.5, "treatment": 0.5},
        )

        self.assertTrue(result["srm_detected"])

    def test_user_metrics_preserve_assigned_users(self):
        users = simulate_users(20, "2026-01-01")
        assignments = assign_users(
            users=users,
            experiment_id="exp_test",
            traffic_allocation={"control": 0.5, "treatment": 0.5},
        )
        events = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "event_name": "started_lesson",
                    "event_timestamp": pd.Timestamp("2026-01-08 01:00:00"),
                    "load_time_ms": None,
                    "amount_usd": None,
                }
            ]
        )

        user_metrics = build_user_metrics(users, assignments, events)

        self.assertEqual(len(user_metrics), len(assignments))
        self.assertEqual(user_metrics["day_7_retained"].sum(), 1)

    def test_experiment_metrics_have_expected_variants(self):
        users = simulate_users(100, "2026-01-01")
        assignments = assign_users(
            users=users,
            experiment_id="exp_test",
            traffic_allocation={"control": 0.5, "treatment": 0.5},
        )
        events = pd.DataFrame(
            columns=[
                "user_id",
                "event_name",
                "event_timestamp",
                "load_time_ms",
                "amount_usd",
            ]
        )
        events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

        user_metrics = build_user_metrics(users, assignments, events)
        experiment_metrics = aggregate_experiment_metrics(user_metrics)

        self.assertEqual(set(experiment_metrics["variant"]), {"control", "treatment"})

    def test_sequential_boundary_is_stricter_early(self):
        early = evaluate_sequential_look(z_statistic=2.2, information_fraction=0.25)
        final = evaluate_sequential_look(z_statistic=2.2, information_fraction=1.0)

        self.assertFalse(early["can_stop_early"])
        self.assertTrue(final["can_stop_early"])


if __name__ == "__main__":
    unittest.main()
