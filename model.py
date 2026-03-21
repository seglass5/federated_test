"""
model.py — Base model loading and LoRA adapter utilities.

Provides functions to load Qwen/Qwen2.5-0.5B, wrap it with a PEFT LoRA
adapter, and serialise/deserialise only the LoRA delta weights for federated
exchange. Only the tiny LoRA adapter weights travel the network; the frozen
base model weights remain on each client machine at all times.
"""

from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import BASE_MODEL_NAME, LORA_ALPHA, LORA_DROPOUT, LORA_R


def load_base_model(device: str) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load the base causal LM and its tokenizer onto *device*.

    Uses float32 for broad CPU compatibility. LoRA is not applied here —
    call apply_lora() on the returned model before training.

    Args:
        device: PyTorch device string, e.g. "cpu" or "cuda".

    Returns:
        (model, tokenizer) both ready for use.
    """
    print(f"[model] Loading {BASE_MODEL_NAME} onto device='{device}' ...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.to(device)
    return model, tokenizer


def apply_lora(model: AutoModelForCausalLM) -> PeftModel:
    """Wrap *model* with a LoRA adapter defined by config.py values.

    Only the q_proj and v_proj attention projections are adapted; all other
    parameters are frozen. The returned PeftModel is ready to be trained.

    Args:
        model: a raw AutoModelForCausalLM (not yet wrapped with PEFT).

    Returns:
        PeftModel with LoRA adapters injected and non-LoRA params frozen.
    """
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def get_lora_parameters(model: PeftModel) -> list[np.ndarray]:
    """Return the LoRA adapter parameters as a list of float32 numpy arrays.

    Only parameters whose names contain 'lora_' are included. The ordering
    is the deterministic insertion order of model.named_parameters() and must
    match the order expected by set_lora_parameters().

    Args:
        model: a PEFT-wrapped model.

    Returns:
        List of numpy arrays, one per LoRA parameter tensor.
    """
    return [
        p.detach().cpu().numpy().astype(np.float32)
        for n, p in model.named_parameters()
        if "lora_" in n
    ]


def set_lora_parameters(model: PeftModel, parameters: list[np.ndarray]) -> None:
    """Overwrite LoRA adapter parameters in-place from *parameters*.

    Parameters must arrive in the same order as returned by
    get_lora_parameters(). Uses .data.copy_() so the autograd graph is
    preserved and gradient tracking metadata is not disrupted.

    Args:
        model: a PEFT-wrapped model whose LoRA weights will be overwritten.
        parameters: list of numpy arrays produced by get_lora_parameters().

    Raises:
        ValueError: if the number of arrays does not match the model.
    """
    lora_named = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    if len(lora_named) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(lora_named)} LoRA "
            f"tensors but received {len(parameters)}."
        )
    with torch.no_grad():
        for (_, param), arr in zip(lora_named, parameters):
            param.data.copy_(torch.from_numpy(arr))


def count_lora_parameters(model: PeftModel) -> dict[str, Any]:
    """Count trainable LoRA parameters versus total model parameters.

    Args:
        model: a PEFT-wrapped model.

    Returns:
        dict with keys:
            "lora_params"  — number of LoRA-only parameters (int)
            "total_params" — total parameter count including frozen base (int)
            "ratio"        — lora_params / total_params (float)
    """
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = lora_params / total_params if total_params > 0 else 0.0
    return {"lora_params": lora_params, "total_params": total_params, "ratio": ratio}
