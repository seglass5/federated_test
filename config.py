"""
config.py — Central configuration for the federated LoRA compliance demo.

All hyperparameters and paths live here so Phase 2 only needs to touch this
file to adjust the training setup.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Federation settings
# ---------------------------------------------------------------------------
NUM_CLIENTS: int = 3
NUM_ROUNDS: int = 5

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
LOCAL_EPOCHS: int = 2

# ---------------------------------------------------------------------------
# LoRA adapter settings
# ---------------------------------------------------------------------------
LORA_R: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
BASE_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LEN: int = 256

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path("data/raw")
PARTITION_DIR: Path = Path("data/partitions")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
