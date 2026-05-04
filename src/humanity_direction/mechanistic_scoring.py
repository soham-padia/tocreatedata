from __future__ import annotations

from pathlib import Path

import torch


def load_direction_vector(path: str | Path, direction_name: str = "global") -> tuple[torch.Tensor, dict]:
    payload = torch.load(Path(path), map_location="cpu")
    if direction_name == "global":
        vector = payload["global"]
    else:
        vector = payload["axes"][direction_name]
    vector = vector.detach().float().cpu()
    norm = vector.norm().clamp_min(1e-12)
    meta = {
        "model_name": payload.get("model_name"),
        "layer_index": payload.get("layer_index"),
        "direction_name": direction_name,
        "vector_norm": float(vector.norm().item()),
    }
    return vector / norm, meta


def projection_score(delta: torch.Tensor, unit_direction: torch.Tensor) -> float:
    return float(torch.dot(delta.detach().float().cpu(), unit_direction).item())
