from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax, title, ylabel=None):
    ax.set_title(title, fontsize=12, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_primary_metric(frequentist_result, output_path):
    output_path = Path(output_path)
    variants = ["Control", "Treatment"]
    rates = [
        frequentist_result["control_rate"] * 100,
        frequentist_result["treatment_rate"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(variants, rates, color=["#5B6770", "#2F80ED"])
    _style_axes(ax, "Day-7 Retention by Variant", "Retention rate (%)")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    lift = frequentist_result["absolute_lift"] * 100
    p_value = frequentist_result["p_value"]
    ax.text(
        0.5,
        max(rates) * 0.72,
        f"Lift: {lift:+.2f}pp\np-value: {p_value:.4f}",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D5DD"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_guardrails(experiment_metrics, output_path):
    output_path = Path(output_path)
    metrics = [
        ("revenue_per_user", "Revenue/user ($)"),
        ("crash_rate", "Crash rate (%)"),
        ("p95_load_time_ms", "P95 load time (ms)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (metric, title) in zip(axes, metrics):
        values = experiment_metrics.set_index("variant")[metric]
        plot_values = values.copy()
        if metric == "crash_rate":
            plot_values = plot_values * 100

        bars = ax.bar(
            ["Control", "Treatment"],
            [plot_values["control"], plot_values["treatment"]],
            color=["#5B6770", "#2F80ED"],
        )
        _style_axes(ax, title)

        for bar in bars:
            value = bar.get_height()
            label = f"{value:.2f}" if metric != "p95_load_time_ms" else f"{value:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(plot_values) * 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Guardrail Metrics by Variant", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_segment_lift(segment_effects, output_path):
    output_path = Path(output_path)
    rows = [
        row
        for row in segment_effects
        if row["segment_column"] in {"platform", "user_segment"}
    ]
    labels = [
        f"{row['segment_column']}={row['segment_value']}\n"
        f"n={row['n_control'] + row['n_treatment']:,}"
        for row in rows
    ]
    lifts = [row["absolute_lift"] * 100 for row in rows]
    colors = ["#2F80ED" if lift >= 0 else "#D64545" for lift in lifts]
    y_positions = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y_positions, lifts, color=colors)
    ax.axvline(0, color="#344054", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    _style_axes(ax, "Day-7 Retention Lift by Segment", "Absolute lift (pp)")

    for y_pos, lift in zip(y_positions, lifts):
        label_x = lift + (0.08 if lift >= 0 else -0.08)
        ax.text(
            label_x,
            y_pos,
            f"{lift:+.2f}pp",
            va="center",
            ha="left" if lift >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_peeking_false_positive_rates(peeking_result, output_path):
    output_path = Path(output_path)
    labels = ["Final-only\nanalysis", "Daily peeking\nstop if p < 0.05"]
    rates = [
        peeking_result["false_positive_rate_final_only"] * 100,
        peeking_result["false_positive_rate_with_peeking"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, rates, color=["#5B6770", "#D64545"])
    _style_axes(ax, "False Positive Inflation from Peeking", "False positive rate (%)")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.text(
        0.5,
        max(rates) * 0.55,
        f"Inflation: {peeking_result['inflation_factor']:.1f}x",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D5DD"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
