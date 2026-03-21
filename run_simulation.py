"""
run_simulation.py — Single entrypoint to run the federated LoRA simulation.

Usage:
    python run_simulation.py

This starts a local Flower simulation with NUM_CLIENTS virtual clients running
NUM_ROUNDS of federated averaging. Ray is used for process-level isolation so
each client runs in its own subprocess, mirroring a real multi-institution setup.

Phase 1: clients return stub parameters and metrics.
Phase 2: clients will perform real LoRA fine-tuning before returning weights.
"""

import flwr as fl

from client import make_client
from config import NUM_CLIENTS, NUM_ROUNDS
from server import make_strategy


def client_fn(cid: str) -> fl.client.NumPyClient:
    """Instantiate a client for the given client ID string provided by Flower."""
    return make_client(int(cid))


def main() -> None:
    print("=" * 60)
    print("Federated Compliance LoRA — Phase 1 Simulation")
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

    # Surface final aggregated metrics if available
    if history.metrics_distributed:
        print("\nAggregated metrics per round:")
        for metric_name, round_values in history.metrics_distributed.items():
            for rnd, value in round_values:
                print(f"  Round {rnd:>2} | {metric_name}: {value:.6f}")


if __name__ == "__main__":
    main()
