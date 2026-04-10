from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class CandidateResult:
    phrase: str
    score: float
    depth: int


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def beam_search_phrases(
    seed_terms: list[str],
    scorer: Callable[[str], float],
    beam_width: int = 8,
    max_phrase_len: int = 3,
    min_improvement: float = 0.0,
) -> list[CandidateResult]:
    seed_terms = _dedupe_preserve_order(seed_terms)
    frontier = [CandidateResult(phrase=term, score=scorer(term), depth=1) for term in seed_terms]
    frontier.sort(key=lambda item: item.score, reverse=True)
    beam = frontier[:beam_width]
    best_by_phrase = {item.phrase: item for item in beam}

    for depth in range(2, max_phrase_len + 1):
        proposals: list[CandidateResult] = []
        for parent in beam:
            for term in seed_terms:
                phrase = f"{parent.phrase} | {term}"
                score = scorer(phrase)
                if score < parent.score + min_improvement:
                    continue
                proposals.append(CandidateResult(phrase=phrase, score=score, depth=depth))

        if not proposals:
            break

        proposals.sort(key=lambda item: item.score, reverse=True)
        beam = proposals[:beam_width]
        for item in beam:
            current = best_by_phrase.get(item.phrase)
            if current is None or item.score > current.score:
                best_by_phrase[item.phrase] = item

    return sorted(best_by_phrase.values(), key=lambda item: item.score, reverse=True)
