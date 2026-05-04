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


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def batch_terminal_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer_index: int,
    device: str,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[layer_index]
    lengths = attention_mask.sum(dim=1) - 1
    gathered = hidden_states[torch.arange(hidden_states.shape[0], device=device), lengths]
    return gathered.detach().float().cpu()


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


@torch.no_grad()
def batch_mean_suffix_activation_from_token_ids(
    model: Any,
    tokenizer: Any,
    token_id_sequences: list[list[int]],
    suffix_lengths: list[int],
    layer_index: int,
    device: str,
) -> torch.Tensor:
    if len(token_id_sequences) != len(suffix_lengths):
        raise ValueError("token_id_sequences and suffix_lengths must have the same length")
    if not token_id_sequences:
        raise ValueError("token_id_sequences must be non-empty")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before batching token ids")

    max_len = max(len(seq) for seq in token_id_sequences)
    input_ids = torch.full(
        (len(token_id_sequences), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(token_id_sequences), max_len),
        dtype=torch.long,
        device=device,
    )

    for row_index, sequence in enumerate(token_id_sequences):
        seq_len = len(sequence)
        if seq_len == 0:
            raise ValueError("each token sequence must be non-empty")
        input_ids[row_index, :seq_len] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row_index, :seq_len] = 1

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[layer_index]

    pooled_rows: list[torch.Tensor] = []
    for row_index, suffix_len in enumerate(suffix_lengths):
        seq_len = int(attention_mask[row_index].sum().item())
        if suffix_len <= 0 or suffix_len > seq_len:
            raise ValueError("invalid suffix length for pooled activation extraction")
        start = seq_len - suffix_len
        pooled_rows.append(hidden_states[row_index, start:seq_len].mean(dim=0))

    return torch.stack(pooled_rows).detach().float().cpu()
