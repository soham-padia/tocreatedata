from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MiningConfig:
    model_name: str
    direction_file: str
    prompts_file: str
    lexicon_file: str
    output_file: str
    dataset_file: str
    beam_width: int = 8
    max_phrase_len: int = 3
    top_k: int = 20
    max_new_tokens: int = 128
    temperature: float = 0.0
