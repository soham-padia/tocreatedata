from __future__ import annotations

from typing import Any

import torch


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def normalized_logprob(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    device: str,
) -> float:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")

    input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    labels[:, : prompt_ids["input_ids"].shape[1]] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    target_len = target_ids["input_ids"].shape[1]
    if target_len == 0:
        return 0.0
    return float(-outputs.loss.item() * target_len / target_len)


@torch.no_grad()
def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    do_sample = temperature > 0
    output = model.generate(
        **encoded,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
