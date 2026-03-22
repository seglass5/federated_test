"""
train_local.py — Train isolated local LoRA adapters for each client.

Produces one adapter per client as a baseline for comparison against the
federated adapter.  Each client is trained for NUM_ROUNDS * LOCAL_EPOCHS
epochs, giving exactly the same gradient-update budget as the federated
condition (NUM_ROUNDS rounds × LOCAL_EPOCHS local epochs per round) without
any cross-client weight sharing.

This compute-budget equivalence is essential for a fair comparison:
any accuracy difference between the local and federated adapters reflects
the benefit of federation, not extra training time.

Saved artefacts (one per client):
    results/local_adapter_client_{i}.npz

Usage:
    python train_local.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
)
from data.dataset import load_client_dataset
from model import apply_lora, get_lora_parameters, load_base_model

RESULTS_DIR: Path = Path("results")
# Total epochs = same number of gradient updates as the federated condition
TOTAL_EPOCHS: int = NUM_ROUNDS * LOCAL_EPOCHS


# ---------------------------------------------------------------------------
# Per-client training
# ---------------------------------------------------------------------------


def train_local_client(client_id: int, device: str = DEVICE) -> None:
    """Train a LoRA adapter on client *client_id*'s local partition.

    The adapter is fine-tuned for TOTAL_EPOCHS epochs using AdamW, then
    saved to results/local_adapter_client_{client_id}.npz.

    Args:
        client_id: integer index of the client whose data to use.
        device: PyTorch device string (default from config.DEVICE).
    """
    print(f"\n{'='*60}")
    print(f"[local] Training client {client_id}  ({TOTAL_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Load model and data
    base_model, tokenizer = load_base_model(device)
    peft_model = apply_lora(base_model)
    peft_model.train()

    dataset = load_client_dataset(client_id, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[local] client {client_id}: {len(dataset)} examples, "
          f"batch_size={BATCH_SIZE}, total_epochs={TOTAL_EPOCHS}")

    # Restrict optimiser to LoRA parameters only
    lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    optimizer = AdamW(lora_params, lr=LEARNING_RATE)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss: torch.Tensor = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        print(
            f"[local] client {client_id}  epoch {epoch:>2}/{TOTAL_EPOCHS}"
            f"  loss={mean_loss:.4f}"
        )

    # Save adapter weights
    arrays = get_lora_parameters(peft_model)
    out_path = RESULTS_DIR / f"local_adapter_client_{client_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{f"arr_{i:04d}": arr for i, arr in enumerate(arrays)})
    print(f"[local] client {client_id}: adapter saved → {out_path}")

    # Free memory before the next client
    del peft_model, base_model, dataset, loader, optimizer
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Local LoRA Training — Phase 3 Baseline")
    print(f"  Clients      : {NUM_CLIENTS}")
    print(f"  Total epochs : {TOTAL_EPOCHS}  "
          f"(= {NUM_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs)")
    print("=" * 60)

    for client_id in range(NUM_CLIENTS):
        train_local_client(client_id)

    print("\nAll local adapters saved to results/")


if __name__ == "__main__":
    main()
