from __future__ import annotations

from typing import Any

import torch


def load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def mean_completion_activation(
    model: Any,
    tokenizer: Any,
    prefix_text: str,
    completion_text: str,
    layer_index: int,
    device: str,
) -> torch.Tensor:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")
    full_ids = tokenizer(f"{prefix_text}{completion_text.strip()}", add_special_tokens=False, return_tensors="pt")

    input_ids = full_ids["input_ids"].to(device)
    attention_mask = full_ids["attention_mask"].to(device)
    prefix_len = int(prefix_ids["input_ids"].shape[1])
    total_len = int(input_ids.shape[1])
    if prefix_len >= total_len:
        raise ValueError("completion produced no tokens after tokenization")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[layer_index][0]
    completion_states = hidden_states[prefix_len:total_len]
    return completion_states.mean(dim=0).detach().float().cpu()
