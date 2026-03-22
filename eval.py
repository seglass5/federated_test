"""
eval.py — Evaluation harness comparing federated vs local LoRA adapters.

Evaluates five conditions against the shared held-out test set using
next-token prediction accuracy.  The eval prompt is presented to the model
without the label; whichever of {"negative", "neutral", "positive"} receives
the highest logit at the final token position is taken as the prediction.

Conditions evaluated:
    1. Base model       — frozen Qwen2.5-0.5B, no adapter
    2. Local A          — base + client_0 local adapter
    3. Local B          — base + client_1 local adapter
    4. Local C          — base + client_2 local adapter
    5. Federated        — base + FedAvg-aggregated adapter

Results are saved to results/eval_results.csv and printed as a table.

Usage:
    python eval.py
"""

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
from transformers import PreTrainedTokenizerBase

from config import DEVICE, MAX_SEQ_LEN, PARTITION_DIR
from data.dataset import LABEL_WORDS
from model import apply_lora, load_base_model, set_lora_parameters

RESULTS_DIR: Path = Path("results")

# Eval prompt excludes the label so we can read the model's next-token logit
_EVAL_PROMPT_TEMPLATE: str = (
    "Sentiment of the following financial statement?\n"
    "{text}\n"
    "Sentiment:"
)

# ANSI codes for bolding the federated row when printing to a TTY
_BOLD: str = "\033[1m"
_RESET: str = "\033[0m"

# Batch size for inference (larger than train batch is fine — no grad storage)
_INFER_BATCH: int = 16


# ---------------------------------------------------------------------------
# Adapter I/O
# ---------------------------------------------------------------------------


def load_adapter_from_npz(model: Any, path: Path) -> Any:
    """Load a saved .npz adapter archive and write the weights into *model*.

    Arrays are loaded in sorted key order (``arr_0000``, ``arr_0001``, …),
    which matches the insertion order used by ``_save_adapter_npz`` in
    server.py and train_local.py.

    Args:
        model: a PeftModel whose LoRA parameters will be overwritten.
        path: path to the .npz file written by the training scripts.

    Returns:
        The same model with updated LoRA weights (modified in-place).

    Raises:
        FileNotFoundError: if *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Adapter file not found: {path}\n"
            "Run `python run_simulation.py` (for federated) or "
            "`python train_local.py` (for local) first."
        )
    npz = np.load(path)
    # sorted() of "arr_0000"…"arr_NNNN" is numerically correct for ≤9999 layers
    arrays = [npz[k] for k in sorted(npz.files)]
    set_lora_parameters(model, arrays)
    return model


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _get_label_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[int, int]:
    """Return a mapping from label int → first-token ID for each label word.

    The model predicts the continuation after "Sentiment:", which in the
    training data is " negative", " neutral", or " positive" (with a leading
    space produced by the tokenizer's BPE merge).  We use a leading space so
    the token IDs match what the model saw during training.

    Args:
        tokenizer: the tokenizer associated with the base model.

    Returns:
        dict mapping {0: neg_tok, 1: neu_tok, 2: pos_tok}.
    """
    label_token_ids: dict[int, int] = {}
    for label_id, word in LABEL_WORDS.items():
        # " word" matches the tokenisation after "Sentiment: " in training
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        label_token_ids[label_id] = tokens[0]
    return label_token_ids


def evaluate_condition(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    test_records: list[dict[str, Any]],
    condition_name: str,
) -> dict[str, Any]:
    """Evaluate *model* on *test_records* using next-token prediction accuracy.

    For each record the eval prompt (without the label) is tokenised and a
    single forward pass is run.  The logit at the final token position is
    inspected for the three label token IDs; the argmax is the prediction.
    Inference runs in batches of _INFER_BATCH for efficiency.

    Args:
        model: AutoModelForCausalLM or PeftModel (both support the same
            forward interface).
        tokenizer: the matching tokenizer.
        test_records: list of {"text": str, "label": int} dicts.
        condition_name: human-readable label for this condition (e.g.
            "Federated") used in the returned dict and printed progress line.

    Returns:
        dict with keys: condition, accuracy, f1_macro, f1_negative,
        f1_neutral, f1_positive, n_examples.
    """
    model.eval()
    label_token_ids = _get_label_token_ids(tokenizer)
    label_ids_ordered = [0, 1, 2]
    token_ids_ordered = [label_token_ids[l] for l in label_ids_ordered]

    prompts = [
        _EVAL_PROMPT_TEMPLATE.format(text=r["text"]) for r in test_records
    ]
    y_true = [r["label"] for r in test_records]
    y_pred: list[int] = []

    # Batch inference — left-padding keeps real tokens at the right edge so
    # logits at position[-1] always correspond to the last real token.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for start in range(0, len(prompts), _INFER_BATCH):
            batch_prompts = prompts[start : start + _INFER_BATCH]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                # -1 leaves room for the predicted token in the context window
                max_length=MAX_SEQ_LEN - 1,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            # outputs.logits: (batch, seq_len, vocab_size)
            # With left-padding, last position is always a real content token
            last_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Select logits for the three label tokens
            tok_tensor = torch.tensor(token_ids_ordered, dtype=torch.long)
            label_logits = last_logits[:, tok_tensor]  # (batch, 3)
            preds = label_logits.argmax(dim=-1).tolist()
            y_pred.extend(label_ids_ordered[p] for p in preds)

    tokenizer.padding_side = original_padding_side

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )

    result = {
        "condition": condition_name,
        "accuracy": acc,
        "f1_macro": float(f1_macro),
        "f1_negative": float(f1_per_class[0]),
        "f1_neutral": float(f1_per_class[1]),
        "f1_positive": float(f1_per_class[2]),
        "n_examples": len(test_records),
    }
    print(
        f"  {condition_name:<14}  acc={acc:.3f}  f1_macro={float(f1_macro):.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_test_records() -> list[dict[str, Any]]:
    """Read the shared test split from disk."""
    path = PARTITION_DIR / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found at {path}. Run `python data/partition.py` first."
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_eval() -> pd.DataFrame:
    """Run all five evaluation conditions and return results as a DataFrame.

    Loads the base model once and reuses it across all conditions:
      - Base model condition uses the raw model before LoRA is applied.
      - Adapter conditions reuse the same PeftModel wrapper, swapping weights
        via load_adapter_from_npz() between conditions.

    Saves results/eval_results.csv and prints a formatted table.

    Returns:
        DataFrame with one row per condition and metric columns.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_records = _load_test_records()
    print(f"[eval] Loaded {len(test_records)} test examples.")

    # Load base model and tokenizer — kept alive for all conditions
    base_model, tokenizer = load_base_model(DEVICE)
    results: list[dict[str, Any]] = []

    print("\n[eval] Running conditions:")

    # ------------------------------------------------------------------
    # 1. Base model — evaluate BEFORE apply_lora modifies the modules
    # ------------------------------------------------------------------
    results.append(
        evaluate_condition(base_model, tokenizer, test_records, "Base model")
    )

    # ------------------------------------------------------------------
    # 2–5. Adapter conditions — apply LoRA once, swap weights each time
    # ------------------------------------------------------------------
    peft_model = apply_lora(base_model)

    adapter_conditions: list[tuple[str, Path]] = [
        ("Local A", RESULTS_DIR / "local_adapter_client_0.npz"),
        ("Local B", RESULTS_DIR / "local_adapter_client_1.npz"),
        ("Local C", RESULTS_DIR / "local_adapter_client_2.npz"),
        ("Federated", RESULTS_DIR / "federated_adapter.npz"),
    ]

    for condition_name, adapter_path in adapter_conditions:
        load_adapter_from_npz(peft_model, adapter_path)
        results.append(
            evaluate_condition(
                peft_model, tokenizer, test_records, condition_name
            )
        )
        gc.collect()

    del peft_model, base_model
    gc.collect()

    # ------------------------------------------------------------------
    # Build DataFrame and save CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[eval] Results saved → {csv_path}")

    # ------------------------------------------------------------------
    # Print formatted table
    # ------------------------------------------------------------------
    display_cols = [
        "condition", "accuracy", "f1_macro",
        "f1_negative", "f1_neutral", "f1_positive",
    ]
    headers = ["Condition", "Accuracy", "F1 Macro", "F1 Neg", "F1 Neu", "F1 Pos"]

    rows: list[list[str]] = []
    for rec in results:
        row = [
            rec["condition"],
            f"{rec['accuracy']:.3f}",
            f"{rec['f1_macro']:.3f}",
            f"{rec['f1_negative']:.3f}",
            f"{rec['f1_neutral']:.3f}",
            f"{rec['f1_positive']:.3f}",
        ]
        rows.append(row)

    # Bold the Federated row when printing to a real terminal
    if sys.stdout.isatty():
        rows = [
            [f"{_BOLD}{cell}{_RESET}" for cell in row]
            if row[0] == "Federated"
            else row
            for row in rows
        ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="simple"))
    print()

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
