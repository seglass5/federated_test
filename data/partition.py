"""
data/partition.py — Download and partition the financial_phrasebank dataset.

Creates a non-IID split across NUM_CLIENTS using Dirichlet allocation
(alpha=0.5) over sentiment labels, simulating realistic data heterogeneity
across financial institutions.

Usage:
    python data/partition.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Allow running from the repo root or from data/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NUM_CLIENTS, PARTITION_DIR, SEED

# Dirichlet concentration parameter — lower = more heterogeneous
DIRICHLET_ALPHA = 0.5

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def download_dataset():
    """Download financial_phrasebank (sentences_allagree split) from HuggingFace."""
    from datasets import load_dataset

    print("Downloading financial_phrasebank (sentences_allagree split)...")
    ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
    # The dataset only has a 'train' split; we carve out our own test set.
    return ds["train"]


def dirichlet_partition(labels: list[int], num_clients: int, alpha: float, rng: np.random.Generator):
    """
    Partition indices into num_clients subsets using Dirichlet allocation
    over the label distribution.

    Returns a list of lists of indices, one per client.
    """
    label_array = np.array(labels)
    unique_labels = np.unique(label_array)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for lbl in unique_labels:
        lbl_indices = np.where(label_array == lbl)[0].tolist()
        rng.shuffle(lbl_indices)

        # Draw proportions from Dirichlet distribution
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

        # Convert proportions to integer counts (last client gets remainder)
        counts = (proportions * len(lbl_indices)).astype(int)
        counts[-1] = len(lbl_indices) - counts[:-1].sum()

        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(lbl_indices[start : start + count])
            start += count

    return client_indices


def print_distribution_table(client_splits: list[list[dict]], test_split: list[dict]):
    """Print a summary table of label counts per client and the test set."""
    label_ids = sorted(LABEL_NAMES.keys())
    col_w = 10

    header = f"{'Client':<12}" + "".join(f"{LABEL_NAMES[l]:>{col_w}}" for l in label_ids) + f"{'Total':>{col_w}}"
    print("\n" + "=" * len(header))
    print("Label distribution across partitions")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, split in enumerate(client_splits):
        counts = Counter(rec["label"] for rec in split)
        row = f"{'client_' + str(i):<12}" + "".join(f"{counts.get(l, 0):>{col_w}}" for l in label_ids)
        row += f"{len(split):>{col_w}}"
        print(row)

    test_counts = Counter(rec["label"] for rec in test_split)
    print("-" * len(header))
    test_row = f"{'test':.<12}" + "".join(f"{test_counts.get(l, 0):>{col_w}}" for l in label_ids)
    test_row += f"{len(test_split):>{col_w}}"
    print(test_row)
    print("=" * len(header))
    print(f"\nDirichlet alpha={DIRICHLET_ALPHA}  (lower = more heterogeneous)\n")


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):>5} records -> {path}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # 1. Download
    dataset = download_dataset()
    all_records = [{"text": ex["sentence"], "label": ex["label"]} for ex in dataset]

    # 2. Hold out ~15% as a shared test set (stratified)
    labels = [rec["label"] for rec in all_records]
    label_array = np.array(labels)
    test_indices = []
    train_indices = []
    for lbl in np.unique(label_array):
        lbl_idx = np.where(label_array == lbl)[0].tolist()
        rng.shuffle(lbl_idx)
        n_test = max(1, int(0.15 * len(lbl_idx)))
        test_indices.extend(lbl_idx[:n_test])
        train_indices.extend(lbl_idx[n_test:])

    test_split = [all_records[i] for i in test_indices]
    train_records = [all_records[i] for i in train_indices]

    # 3. Non-IID Dirichlet partition of train set across clients
    train_labels = [rec["label"] for rec in train_records]
    client_index_groups = dirichlet_partition(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA, rng)
    client_splits = [[train_records[i] for i in idxs] for idxs in client_index_groups]

    # 4. Print distribution table
    print_distribution_table(client_splits, test_split)

    # 5. Save partitions
    print("Saving partitions...")
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    for i, split in enumerate(client_splits):
        save_jsonl(split, PARTITION_DIR / f"client_{i}" / "train.jsonl")
    save_jsonl(test_split, PARTITION_DIR / "test.jsonl")

    print("\nPartitioning complete.")


if __name__ == "__main__":
    main()
