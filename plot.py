"""
plot.py — Produce visualisations for the Phase 3 results.

Generates two figures and saves them to results/:

    results/accuracy_comparison.png
        Horizontal bar chart comparing accuracy across all five conditions.
        The Federated bar is highlighted in teal; others in amber.

    results/loss_curve.png
        Line chart of aggregated train_loss and eval_loss per FL round,
        read from results/round_metrics.csv (written by server.py).

Usage:
    python plot.py
"""

from pathlib import Path

import matplotlib
# Non-interactive backend — safe for headless / CI environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use)
import pandas as pd              # noqa: E402

RESULTS_DIR: Path = Path("results")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_TEAL: str = "#2A9D8F"   # federated condition
_AMBER: str = "#E9C46A"  # local / base conditions


# ---------------------------------------------------------------------------
# Figure 1 — Accuracy comparison
# ---------------------------------------------------------------------------


def plot_accuracy_comparison(eval_csv: Path, out_path: Path) -> None:
    """Horizontal bar chart of accuracy for each evaluation condition.

    Args:
        eval_csv: path to results/eval_results.csv produced by eval.py.
        out_path: destination PNG path.

    Raises:
        FileNotFoundError: if *eval_csv* does not exist.
    """
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run `python eval.py` first."
        )
    df = pd.read_csv(eval_csv)

    conditions = df["condition"].tolist()
    accuracies = df["accuracy"].tolist()
    colours = [_TEAL if c == "Federated" else _AMBER for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(conditions, accuracies, color=colours, height=0.55, edgecolor="white")

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by training condition", fontsize=13, fontweight="bold", pad=12)

    # Light vertical gridlines; remove top and right spines
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: teal = federated, amber = local / base
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_TEAL, label="Federated"),
        Patch(facecolor=_AMBER, label="Local / Base"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Federated loss curve
# ---------------------------------------------------------------------------


def plot_loss_curve(metrics_csv: Path, out_path: Path) -> None:
    """Line chart of aggregated train_loss and eval_loss per FL round.

    Args:
        metrics_csv: path to results/round_metrics.csv written by server.py.
        out_path: destination PNG path.

    Raises:
        FileNotFoundError: if *metrics_csv* does not exist.
    """
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Round metrics not found at {metrics_csv}. "
            "Run `python run_simulation.py` first."
        )
    df = pd.read_csv(metrics_csv)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        df["round"],
        df["train_loss"],
        marker="o",
        color=_TEAL,
        linewidth=2,
        markersize=6,
        label="Train loss (aggregated)",
    )
    ax.plot(
        df["round"],
        df["eval_loss"],
        marker="s",
        color=_AMBER,
        linewidth=2,
        markersize=6,
        linestyle="--",
        label="Eval loss (aggregated)",
    )

    ax.set_xlabel("Federated round", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title(
        "Federated training loss over rounds",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(df["round"].tolist())

    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(
        eval_csv=RESULTS_DIR / "eval_results.csv",
        out_path=RESULTS_DIR / "accuracy_comparison.png",
    )
    plot_loss_curve(
        metrics_csv=RESULTS_DIR / "round_metrics.csv",
        out_path=RESULTS_DIR / "loss_curve.png",
    )

    print(f"\n[plot] Both figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
