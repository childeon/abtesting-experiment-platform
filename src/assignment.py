import hashlib

import pandas as pd


def hash_to_bucket(user_id, experiment_id):
    hash_input = f"{experiment_id}:{user_id}".encode()
    digest = hashlib.md5(hash_input).hexdigest()
    hash_value = int(digest, 16)
    max_hash_value = 16**32 - 1
    return hash_value / max_hash_value


def validate_traffic_allocation(traffic_allocation):
    if not traffic_allocation:
        raise ValueError("traffic_allocation cannot be empty.")

    total = sum(traffic_allocation.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("traffic_allocation values must sum to 1.")

    for variant, allocation in traffic_allocation.items():
        if allocation < 0:
            raise ValueError(f"Allocation for {variant} cannot be negative.")


def assign_variant(user_id, experiment_id, traffic_allocation):
    validate_traffic_allocation(traffic_allocation)

    hash_bucket = hash_to_bucket(user_id, experiment_id)
    cumulative_allocation = 0.0

    for variant, allocation in traffic_allocation.items():
        cumulative_allocation += allocation
        if hash_bucket < cumulative_allocation:
            return {
                "variant": variant,
                "hash_bucket": hash_bucket,
            }

    final_variant = list(traffic_allocation.keys())[-1]
    return {
        "variant": final_variant,
        "hash_bucket": hash_bucket,
    }


def assign_users(users, experiment_id, traffic_allocation):
    required_columns = {"user_id", "signup_timestamp"}
    missing_columns = required_columns - set(users.columns)
    if missing_columns:
        raise ValueError(f"users is missing required columns: {missing_columns}")

    assignments = users[["user_id", "signup_timestamp"]].copy()
    assignment_results = assignments["user_id"].apply(
        lambda user_id: assign_variant(
            user_id=user_id,
            experiment_id=experiment_id,
            traffic_allocation=traffic_allocation,
        )
    )

    assignments["experiment_id"] = experiment_id
    assignments["variant"] = assignment_results.apply(lambda result: result["variant"])
    assignments["hash_bucket"] = assignment_results.apply(
        lambda result: result["hash_bucket"]
    )
    assignments = assignments.rename(columns={"signup_timestamp": "assigned_at"})

    return assignments[
        ["user_id", "experiment_id", "variant", "assigned_at", "hash_bucket"]
    ]


def summarize_assignment_balance(assignments):
    balance = assignments["variant"].value_counts(normalize=True).reset_index()
    balance.columns = ["variant", "assignment_share"]
    return balance
