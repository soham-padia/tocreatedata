from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass(slots=True)
class DirectionAxis:
    name: str
    weight: float
    positive_cues: list[str]
    negative_cues: list[str]


@dataclass(slots=True)
class DirectionSpec:
    name: str
    description: str
    axes: list[DirectionAxis]


@dataclass(slots=True)
class AxisScore:
    axis: str
    positive_hits: int
    negative_hits: int
    raw_score: float
    weighted_score: float


@dataclass(slots=True)
class DirectionScore:
    total: float
    breakdown: list[AxisScore]


def load_direction_spec(path: str | Path) -> DirectionSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    axes = [
        DirectionAxis(
            name=item["name"],
            weight=float(item.get("weight", 1.0)),
            positive_cues=list(item.get("positive_cues", [])),
            negative_cues=list(item.get("negative_cues", [])),
        )
        for item in payload["axes"]
    ]
    return DirectionSpec(
        name=payload["name"],
        description=payload.get("description", ""),
        axes=axes,
    )


def score_text_against_direction(text: str, direction: DirectionSpec) -> DirectionScore:
    normalized = _normalize(text)
    breakdown: list[AxisScore] = []
    total = 0.0
    for axis in direction.axes:
        positive_hits = sum(1 for cue in axis.positive_cues if _normalize(cue) in normalized)
        negative_hits = sum(1 for cue in axis.negative_cues if _normalize(cue) in normalized)
        raw_score = float(positive_hits - negative_hits)
        weighted_score = raw_score * axis.weight
        breakdown.append(
            AxisScore(
                axis=axis.name,
                positive_hits=positive_hits,
                negative_hits=negative_hits,
                raw_score=raw_score,
                weighted_score=weighted_score,
            )
        )
        total += weighted_score
    return DirectionScore(total=total, breakdown=breakdown)
