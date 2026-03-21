"""
client.py — Flower client definition for federated LoRA fine-tuning.

Each FlowerClient represents one financial institution. In Phase 1 the fit
and evaluate methods are stubs that return realistic-looking metrics without
performing real training. Phase 2 will replace the stubs with actual LoRA
fine-tuning via PEFT + Transformers.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

from config import LOCAL_EPOCHS, PARTITION_DIR


class FlowerClient(fl.client.NumPyClient):
    """Federated learning client representing a single financial institution."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.dataset = self._load_partition()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_partition(self) -> list[dict]:
        """Load this client's training partition from disk."""
        partition_path = PARTITION_DIR / f"client_{self.client_id}" / "train.jsonl"
        if not partition_path.exists():
            raise FileNotFoundError(
                f"Partition not found at {partition_path}. "
                "Run `python data/partition.py` first."
            )
        with partition_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"[client {self.client_id}] Loaded {len(records)} training examples.")
        return records

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current (stub) LoRA adapter parameters as numpy arrays.

        Phase 2: replace with actual PEFT model parameter extraction.
        """
        # Return a single zero-filled array as a stand-in for LoRA weights
        return [np.zeros(1, dtype=np.float32)]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Apply received (aggregated) parameters to the local model.

        Phase 2: deserialize and load into the PEFT adapter.
        """
        # No-op in Phase 1
        pass

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Fine-tune the LoRA adapter on local data and return updated weights.

        Returns:
            parameters: updated adapter weights (unchanged in Phase 1 stub)
            num_examples: size of the local training set
            metrics: dict with training metrics for aggregation
        """
        self.set_parameters(parameters)

        # TODO (Phase 2): Replace this stub with real LoRA fine-tuning.
        #   Steps:
        #     1. Load BASE_MODEL_NAME with AutoModelForCausalLM / AutoModelForSequenceClassification
        #     2. Wrap with peft.get_peft_model using LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        #        lora_dropout=LORA_DROPOUT, target_modules=[...])
        #     3. Tokenize self.dataset with AutoTokenizer
        #     4. Run LOCAL_EPOCHS of gradient descent (e.g. via HuggingFace Trainer or manual loop)
        #     5. Extract only the trainable LoRA delta weights with get_peft_model_state_dict()
        #     6. Return those weights as NDArrays below

        stub_loss = 1.0
        num_examples = len(self.dataset)

        print(
            f"[client {self.client_id}] fit() — "
            f"epochs={LOCAL_EPOCHS}, examples={num_examples}, loss={stub_loss:.4f} (stub)"
        )

        return self.get_parameters(config={}), num_examples, {"loss": stub_loss}

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the model on the local partition's held-out portion.

        Returns:
            loss: scalar loss value
            num_examples: number of evaluation examples
            metrics: dict with evaluation metrics
        """
        self.set_parameters(parameters)

        # TODO (Phase 2): Run real inference over a local eval split and compute
        # cross-entropy loss and accuracy.

        stub_loss = 1.0
        num_examples = len(self.dataset)
        stub_accuracy = 0.0

        print(
            f"[client {self.client_id}] evaluate() — "
            f"examples={num_examples}, loss={stub_loss:.4f}, "
            f"accuracy={stub_accuracy:.4f} (stub)"
        )

        return stub_loss, num_examples, {"accuracy": stub_accuracy}


# ---------------------------------------------------------------------------
# Factory function used by the simulation entrypoint
# ---------------------------------------------------------------------------


def make_client(client_id: int) -> FlowerClient:
    """Instantiate and return a FlowerClient for the given client ID."""
    return FlowerClient(client_id=client_id)
