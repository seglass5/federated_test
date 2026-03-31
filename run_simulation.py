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

Memory note:
    Each virtual client loads Qwen/Qwen2.5-0.5B independently.  In float32 the
    model plus gradients and optimiser state peak at ~7–10 GB per client process.
    To prevent OOM the simulation runs clients sequentially: client_resources is
    set to {"num_cpus": NUM_CLIENTS} so Ray can only schedule one client at a
    time given the NUM_CLIENTS total CPUs allocated to the Ray instance.  This
    keeps peak memory at one model's worth regardless of NUM_CLIENTS.
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
        # Each client claims all available CPUs → only one runs at a time.
        # This prevents the ~7–10 GB per-client peak from stacking across
        # all NUM_CLIENTS processes simultaneously and causing OOM.
        client_resources={"num_cpus": NUM_CLIENTS, "num_gpus": 0.0},
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
