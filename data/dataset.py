"""
data/dataset.py — PyTorch Dataset for the compliance sentiment fine-tuning task.

Formats financial_phrasebank records with a fixed prompt template and tokenises
them for causal language modelling.  The model learns to predict the sentiment
label token given the prompt, so no classification head is needed and parameter
serialisation for federated exchange is straightforward.

Prompt template:
    Sentiment of the following financial statement?
    {text}
    Sentiment: {label_word}

Usage:
    from data.dataset import load_client_dataset, load_test_dataset
"""

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Allow importing config whether this module is imported from the repo root
# (normal use) or run directly from data/ (standalone testing).
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_SEQ_LEN, PARTITION_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_WORDS: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

_PROMPT_TEMPLATE: str = (
    "Sentiment of the following financial statement?\n"
    "{text}\n"
    "Sentiment: {label_word}\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_example(record: dict[str, Any]) -> str:
    """Format one dataset record as a complete prompt+completion string."""
    label_word = LABEL_WORDS[record["label"]]
    return _PROMPT_TEMPLATE.format(text=record["text"], label_word=label_word)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl file and return its records as a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. "
            "Run `python data/partition.py` first."
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class ComplianceDataset(Dataset):
    """Tokenised dataset for causal LM fine-tuning on financial sentiment.

    Each item is a dict of three tensors:
        "input_ids"      — token ids, shape (MAX_SEQ_LEN,)
        "attention_mask" — 1 for real tokens, 0 for padding, same shape
        "labels"         — copy of input_ids with padding positions set to -100
                           so CrossEntropyLoss ignores them automatically

    All tensors are produced once in the constructor and stored in memory,
    which is practical for the ~1 000-example partitions in this demo.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._encodings = self._tokenize(records, tokenizer)

    # ------------------------------------------------------------------

    def _tokenize(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]:
        """Batch-tokenise all records and build the labels tensor."""
        texts = [_format_example(r) for r in records]
        enc = tokenizer(
            texts,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Clone input_ids into labels then mask padding with -100
        labels: torch.Tensor = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
            "labels": self._encodings["labels"][idx],
        }


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------


def load_client_dataset(
    client_id: int,
    tokenizer: PreTrainedTokenizerBase,
) -> ComplianceDataset:
    """Load and tokenise the training partition for *client_id*.

    Args:
        client_id: integer client index matching a data/partitions/client_N/
            directory created by data/partition.py.
        tokenizer: the tokenizer associated with the base model.

    Returns:
        ComplianceDataset ready for use with a DataLoader.
    """
    path = PARTITION_DIR / f"client_{client_id}" / "train.jsonl"
    records = _read_jsonl(path)
    return ComplianceDataset(records, tokenizer)


def load_test_dataset(tokenizer: PreTrainedTokenizerBase) -> ComplianceDataset:
    """Load and tokenise the shared held-out test set.

    Args:
        tokenizer: the tokenizer associated with the base model.

    Returns:
        ComplianceDataset ready for evaluation.
    """
    path = PARTITION_DIR / "test.jsonl"
    records = _read_jsonl(path)
    return ComplianceDataset(records, tokenizer)
