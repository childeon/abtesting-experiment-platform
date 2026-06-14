import numpy as np
import pandas as pd


def simulate_users(n_users, start_date, seed=42):
    if n_users <= 0:
        raise ValueError("n_users must be positive.")

    rng = np.random.default_rng(seed)

    users = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "signup_timestamp": pd.date_range(
                start=start_date,
                periods=n_users,
                freq="15min",
            ),
            "country": rng.choice(
                ["US", "UK", "IN", "BR"],
                size=n_users,
                p=[0.50, 0.20, 0.20, 0.10],
            ),
            "platform": rng.choice(
                ["web", "ios", "android"],
                size=n_users,
                p=[0.30, 0.50, 0.20],
            ),
            "user_segment": rng.choice(
                ["low_intent", "medium_intent", "high_intent"],
                size=n_users,
                p=[0.50, 0.35, 0.15],
            ),
        }
    )

    return users
