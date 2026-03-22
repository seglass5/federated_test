"""
run_simulation.py — Single entrypoint to run the federated LoRA simulation.

Usage:
    python run_simulation.py

This starts a local Flower simulation with NUM_CLIENTS virtual clients running
NUM_ROUNDS of federated averaging. Ray is used for process-level isolation so
each client runs in its own subprocess, mirroring a real multi-institution setup.

After the final round the strategy automatically writes:
    results/federated_adapter.npz  — aggregated LoRA weights
    results/round_metrics.csv      — per-round train_loss and eval_loss

Memory note (Phase 2 / Phase 3):
    Each virtual client loads Qwen/Qwen2.5-0.5B independently (~1 GB float32).
    With NUM_CLIENTS=3 this requires approximately 3 GB of free RAM.  If memory
    is tight, lower NUM_CLIENTS in config.py before running.
"""

from pathlib import Path

import flwr as fl

from client import make_client
from config import NUM_CLIENTS, NUM_ROUNDS
from server import make_strategy

# ---------------------------------------------------------------------------
# Client cache — avoids reloading the model on every Flower callback
# ---------------------------------------------------------------------------
# Flower's simulation engine calls client_fn each time it needs a client
# (once for fit, once for evaluate per round). The Ray actor for a given
# client ID stays alive across calls in the same process, so a module-level
# dict persists the loaded model and tokenizer rather than reloading them.

_client_cache: dict[int, fl.client.NumPyClient] = {}


def client_fn(cid: str) -> fl.client.NumPyClient:
    """Instantiate (or retrieve from cache) the client for *cid*."""
    client_id = int(cid)
    if client_id not in _client_cache:
        _client_cache[client_id] = make_client(client_id)
    return _client_cache[client_id]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Ensure the results directory exists before the strategy tries to write to it
    Path("results").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Federated Compliance LoRA — Phase 2 Simulation")
    print(f"  Clients  : {NUM_CLIENTS}")
    print(f"  Rounds   : {NUM_ROUNDS}")
    print("=" * 60)
    print()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=make_strategy(),
        ray_init_args={"num_cpus": NUM_CLIENTS},
    )

    print()
    print("=" * 60)
    print("Simulation complete — ready for Phase 2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Final summary: show train_loss and eval_loss for every round
    # ------------------------------------------------------------------
    fit_metrics: dict = history.metrics_distributed_fit   # from fit aggregation
    eval_metrics: dict = history.metrics_distributed       # from evaluate aggregation

    train_by_round: dict[int, float] = {
        rnd: val
        for rnd, val in fit_metrics.get("train_loss", [])
    }
    eval_by_round: dict[int, float] = {
        rnd: val
        for rnd, val in eval_metrics.get("eval_loss", [])
    }

    all_rounds = sorted(set(train_by_round) | set(eval_by_round))
    if all_rounds:
        print("\nPer-round aggregated metrics:")
        print(f"  {'Round':>5}  {'train_loss':>12}  {'eval_loss':>12}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*12}")
        for rnd in all_rounds:
            tl = f"{train_by_round[rnd]:.6f}" if rnd in train_by_round else "     —"
            el = f"{eval_by_round[rnd]:.6f}" if rnd in eval_by_round else "     —"
            print(f"  {rnd:>5}  {tl:>12}  {el:>12}")
        print()


if __name__ == "__main__":
    main()
