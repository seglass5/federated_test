"""
server.py — Flower server and FedAvg strategy for federated LoRA fine-tuning.

The server orchestrates NUM_ROUNDS of communication with all clients.
In each round it:
  1. Sends the current global LoRA adapter weights to every client.
  2. Receives locally updated weights back.
  3. Aggregates them via FedAvg (weighted average by num_examples).
  4. Logs per-round train_loss and eval_loss to stdout and results/round_metrics.csv.

After the final round the aggregated LoRA adapter is saved as
results/federated_adapter.npz for downstream evaluation.

Only LoRA delta weights travel the network — the base model weights never
leave any institution.
"""

from pathlib import Path
from typing import Any, Optional, Union

import flwr as fl
import numpy as np
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from config import NUM_CLIENTS, NUM_ROUNDS

# Directory where artefacts are written; mirrored in run_simulation.py
RESULTS_DIR: Path = Path("results")


# ---------------------------------------------------------------------------
# Per-round metrics aggregation functions
# ---------------------------------------------------------------------------


def fit_metrics_aggregation_fn(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from all clients after each round.

    Computes a weighted average of train_loss across clients, weighted by the
    number of examples each client trained on.

    Args:
        metrics: list of (num_examples, metrics_dict) tuples, one per client.

    Returns:
        Aggregated metrics dict with key "train_loss".
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
        Aggregated metrics dict with key "eval_loss".
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
# Adapter serialisation helper
# ---------------------------------------------------------------------------


def _save_adapter_npz(arrays: list[np.ndarray], path: Path) -> None:
    """Save a list of numpy arrays to *path* as a .npz archive.

    Arrays are keyed ``arr_NNNN`` (zero-padded 4-digit index) so that
    lexicographic sort recovers the original insertion order for any
    realistic number of LoRA layers.

    Args:
        arrays: list of float32 numpy arrays (LoRA parameters in order).
        path: destination .npz path; parent directory is created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"arr_{i:04d}": arr for i, arr in enumerate(arrays)})
    print(f"[server] Federated adapter saved → {path}")


# ---------------------------------------------------------------------------
# FedAvg subclass that persists artefacts
# ---------------------------------------------------------------------------


class FedAvgWithSave(FedAvg):
    """FedAvg strategy augmented with per-round CSV logging and adapter saving.

    After each round, aggregated train_loss and eval_loss are appended to
    ``results/round_metrics.csv``.  After the final round the aggregated LoRA
    adapter is written to ``results/federated_adapter.npz``.
    """

    def __init__(self, results_dir: Path, **kwargs: Any) -> None:
        super().__init__(
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs,
        )
        self._results_dir = results_dir
        self._csv_path = results_dir / "round_metrics.csv"
        self._round_train: dict[int, float] = {}
        self._round_eval: dict[int, float] = {}
        # Start every simulation run with a clean CSV
        if self._csv_path.exists():
            self._csv_path.unlink()

    # ------------------------------------------------------------------
    # FedAvg hook overrides
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results; save adapter + update CSV on final round."""
        params, metrics = super().aggregate_fit(server_round, results, failures)

        if params is not None:
            train_loss = metrics.get("train_loss")
            if train_loss is not None:
                self._round_train[server_round] = float(train_loss)

            # Persist the final federated adapter after the last round
            if server_round == NUM_ROUNDS:
                arrays = fl.common.parameters_to_ndarrays(params)
                _save_adapter_npz(
                    arrays, self._results_dir / "federated_adapter.npz"
                )

        self._try_write_csv(server_round)
        return params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluate results; update CSV."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return aggregated  # type: ignore[return-value]

        loss, metrics = aggregated
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            self._round_eval[server_round] = float(eval_loss)

        self._try_write_csv(server_round)
        return loss, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_write_csv(self, server_round: int) -> None:
        """Append a CSV row once both train_loss and eval_loss are available."""
        if (
            server_round not in self._round_train
            or server_round not in self._round_eval
        ):
            return

        write_header = not self._csv_path.exists()
        with self._csv_path.open("a") as fh:
            if write_header:
                fh.write("round,train_loss,eval_loss\n")
            tl = self._round_train[server_round]
            el = self._round_eval[server_round]
            fh.write(f"{server_round},{tl:.6f},{el:.6f}\n")


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def make_strategy() -> FedAvgWithSave:
    """Build and return the saving FedAvg strategy for the simulation."""
    return FedAvgWithSave(
        results_dir=RESULTS_DIR,
        # All clients participate in every round
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
