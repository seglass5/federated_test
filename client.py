"""
client.py — Flower client definition for federated LoRA fine-tuning.

Each FlowerClient represents one financial institution. In fit(), the client
genuinely fine-tunes a LoRA adapter on its local partition using a plain
PyTorch training loop. Only the tiny LoRA delta weights are exchanged with
the server — the frozen base model parameters never leave this process.
"""

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays, Scalar

from config import BATCH_SIZE, DEVICE, LEARNING_RATE, LOCAL_EPOCHS
from model import (
    apply_lora,
    count_lora_parameters,
    get_lora_parameters,
    load_base_model,
    set_lora_parameters,
)
from data.dataset import load_client_dataset


class FlowerClient(fl.client.NumPyClient):
    """Federated learning client representing a single financial institution."""

    def __init__(self, client_id: int, device: str = DEVICE) -> None:
        self.client_id = client_id
        self.device = device

        # Load base model and wrap with LoRA adapter
        base_model, self.tokenizer = load_base_model(device)
        self.model = apply_lora(base_model)
        self.model.train()

        # Report adapter size so the compression ratio is visible in logs
        stats = count_lora_parameters(self.model)
        print(
            f"[client {client_id}] LoRA parameters: "
            f"{stats['lora_params']:,} / {stats['total_params']:,} "
            f"({stats['ratio']:.4%})"
        )

        # Tokenise and store the local training partition
        self.dataset = load_client_dataset(client_id, self.tokenizer)
        print(f"[client {client_id}] Dataset ready — {len(self.dataset)} examples.")

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current LoRA adapter parameters as numpy arrays."""
        return get_lora_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Write received aggregated LoRA parameters back into the model.

        Called at the start of every fit() and evaluate() so the client
        begins from the server's latest global adapter state.
        """
        set_lora_parameters(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Fine-tune the LoRA adapter on local data and return updated weights.

        Steps:
          1. Overwrite adapter weights with the server's aggregated parameters.
          2. Run LOCAL_EPOCHS of AdamW optimisation (LoRA params only).
          3. Return updated adapter weights, dataset size, and mean train loss.

        Returns:
            (updated_parameters, num_examples, {"train_loss": float})
        """
        self.set_parameters(parameters)
        self.model.train()

        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Restrict the optimiser to LoRA parameters only — base model is frozen
        lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]
        optimizer = AdamW(lora_params, lr=LEARNING_RATE)

        total_loss = 0.0
        total_batches = 0

        for _epoch in range(LOCAL_EPOCHS):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss: torch.Tensor = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)
        print(
            f"[client {self.client_id}] fit() — "
            f"epochs={LOCAL_EPOCHS}, examples={len(self.dataset)}, "
            f"train_loss={mean_loss:.4f}"
        )
        return self.get_parameters(config={}), len(self.dataset), {"train_loss": mean_loss}

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Compute mean cross-entropy loss on the local partition.

        The evaluation runs over the same training partition used for fit().
        In a production system this would be a held-out local test split; for
        this demo it gives a consistent per-client loss signal.

        Returns:
            (eval_loss, num_examples, {"eval_loss": float})
        """
        self.set_parameters(parameters)
        self.model.eval()

        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                total_batches += 1

        mean_loss = total_loss / max(total_batches, 1)
        print(
            f"[client {self.client_id}] evaluate() — "
            f"examples={len(self.dataset)}, eval_loss={mean_loss:.4f}"
        )
        return mean_loss, len(self.dataset), {"eval_loss": mean_loss}


# ---------------------------------------------------------------------------
# Factory function used by the simulation entrypoint
# ---------------------------------------------------------------------------


def make_client(client_id: int, device: str = DEVICE) -> FlowerClient:
    """Instantiate and return a FlowerClient for the given client ID."""
    return FlowerClient(client_id=client_id, device=device)
