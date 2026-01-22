"""
Model utilities for LLM activation extraction.

This module provides functions to load transformer models and extract
head-wise activations for polarization analysis.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional


def load_model(
    path: str,
    dtype: Optional[torch.dtype] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal language model and tokenizer.

    Args:
        path: Path to the model directory
        dtype: Optional dtype override (defaults to bfloat16 if supported, else float16)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {path}...")

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)

    # Left padding allows batching without destroying the last token position
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


@torch.no_grad()
def extract_heads_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    system_msg: str,
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """
    Extract attention head activations from the last token position.

    This function processes batches of prompts through the model and extracts
    the pre-projection activations from each attention head at the last token.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer
        texts: List of user messages to process
        system_msg: System message to prepend
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization

    Returns:
        numpy array of shape [N, L, H, D] where:
        - N = number of texts
        - L = number of layers
        - H = number of attention heads
        - D = head dimension
    """
    model.eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    D_head = model.config.hidden_size // H

    activations = []

    # Pre-allocate hook containers
    layer_outputs = [None] * L

    def get_hook(layer_idx):
        def hook(module, input, output):
            # Input[0] shape: [Batch, Seq, Hidden]
            # Reshape to [Batch, Seq, Heads, Head_Dim]
            # Move to CPU immediately to free VRAM for the next batch
            reshaped = input[0].detach().view(input[0].shape[0], input[0].shape[1], H, D_head)
            layer_outputs[layer_idx] = reshaped[:, -1, :, :].float().cpu().numpy()
        return hook

    # Register hooks
    hooks = []
    for li in range(L):
        hooks.append(
            model.model.layers[li].self_attn.o_proj.register_forward_hook(get_hook(li))
        )

    print(f"  > Extracting with Batch Size {batch_size}...")

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Format using chat template
            formatted_batch = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for t in batch
            ]

            enc = tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)

            # Forward pass triggers hooks
            model(**enc)

            # Stack layers: [Batch, Layers, Heads, Dim]
            batch_acts = np.stack(layer_outputs, axis=1)
            activations.append(batch_acts)
    finally:
        # Always remove hooks
        for h in hooks:
            h.remove()

    return np.concatenate(activations, axis=0)


def get_model_info(model: AutoModelForCausalLM) -> dict:
    """
    Get information about the model architecture.

    Args:
        model: The loaded transformer model

    Returns:
        Dictionary with model info (num_layers, num_heads, hidden_size, head_dim)
    """
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "head_dim": config.hidden_size // config.num_attention_heads,
    }
