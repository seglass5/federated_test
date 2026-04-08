"""
config.py — Central configuration for the federated LoRA compliance demo.

All hyperparameters and paths live here; both Phase 1 and Phase 2 source
their settings exclusively from this module.
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
LOCAL_EPOCHS: int = 1   # was 2; halves per-round training time on CPU
BATCH_SIZE: int = 8
LEARNING_RATE: float = 3e-4

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
MAX_SEQ_LEN: int = 128  # was 256; quartered attention compute per batch
# CPU-only; change to "cuda" if a GPU is available (not required for Phase 2)
DEVICE: str = "cpu"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path("data/raw")
PARTITION_DIR: Path = Path("data/partitions")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
