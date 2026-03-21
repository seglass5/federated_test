"""
server.py — Flower server and FedAvg strategy for federated LoRA fine-tuning.

The server orchestrates NUM_ROUNDS of communication with all clients.
In each round it:
  1. Sends the current global LoRA adapter weights to every client.
  2. Receives locally updated weights back.
  3. Aggregates them via FedAvg (weighted average by num_examples).
  4. Logs per-round train_loss and eval_loss metrics.

Only LoRA delta weights travel the network — the base model weights never
leave any institution.
"""

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg

from config import NUM_CLIENTS


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from all clients after each round.

    Computes a weighted average of train_loss across clients, weighted by the
    number of examples each client trained on.

    Args:
        metrics: list of (num_examples, metrics_dict) tuples, one per client.

    Returns:
        Aggregated metrics dict logged by the Flower framework.
    """
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    weighted_loss = sum(n * m["train_loss"] for n, m in metrics if "train_loss" in m)
    avg_train_loss = weighted_loss / total_examples

    print(
        f"\n[server] Round fit complete   — "
        f"avg train_loss: {avg_train_loss:.6f}  (across {len(metrics)} clients)\n"
    )
    return {"train_loss": avg_train_loss}


def evaluate_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluate metrics from all clients after each round.

    Args:
        metrics: list of (num_examples, metrics_dict) tuples, one per client.

    Returns:
        Aggregated metrics dict logged by the Flower framework.
    """
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    weighted_loss = sum(n * m["eval_loss"] for n, m in metrics if "eval_loss" in m)
    avg_eval_loss = weighted_loss / total_examples

    print(
        f"[server] Round eval complete  — "
        f"avg eval_loss:  {avg_eval_loss:.6f}  (across {len(metrics)} clients)"
    )
    return {"eval_loss": avg_eval_loss}


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def make_strategy() -> FedAvg:
    """Build and return the FedAvg strategy for the simulation."""
    strategy = FedAvg(
        # All clients participate in every round
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        # Custom aggregation for metrics logging
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    return strategy
