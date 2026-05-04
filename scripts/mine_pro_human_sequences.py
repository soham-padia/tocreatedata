from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import asdict, dataclass
from itertools import count, product
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.activations import batch_terminal_activations, load_model_and_tokenizer
from humanity_direction.data import load_lines, write_jsonl
from humanity_direction.mechanistic_scoring import load_direction_vector
from humanity_direction.prompting import build_chat_prompt, build_injected_prompt
from humanity_direction.scoring import choose_device


DEFAULT_CARRIER_PROMPT = "Please continue with a helpful response."


@dataclass(slots=True)
class SequenceMiningConfig:
    model_name: str
    direction_tensors: str
    direction_name: str
    direction_sign: float
    lexicon_file: str
    output_dir: str
    carrier_prompt: str = DEFAULT_CARRIER_PROMPT
    search_mode: str = "exhaustive"
    min_phrase_len: int = 1
    max_phrase_len: int = 5
    batch_size: int = 128
    retain_top_k: int = 5000
    beam_width: int = 128
    shard_index: int = 0
    num_shards: int = 1


def parse_args() -> SequenceMiningConfig:
    parser = argparse.ArgumentParser(
        description="Score short lexicon sequences by mechanistic projection onto a saved direction vector."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--direction-tensors", required=True)
    parser.add_argument("--direction-name", default="global")
    parser.add_argument("--direction-sign", type=float, default=1.0)
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--carrier-prompt", default=DEFAULT_CARRIER_PROMPT)
    parser.add_argument("--search-mode", choices=("exhaustive", "beam"), default="exhaustive")
    parser.add_argument("--min-phrase-len", type=int, default=1)
    parser.add_argument("--max-phrase-len", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retain-top-k", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.min_phrase_len < 1:
        parser.error("--min-phrase-len must be >= 1")
    if args.max_phrase_len < args.min_phrase_len:
        parser.error("--max-phrase-len must be >= --min-phrase-len")
    if args.direction_sign == 0:
        parser.error("--direction-sign must be nonzero")
    if args.beam_width < 1:
        parser.error("--beam-width must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    return SequenceMiningConfig(**vars(args))


def sequence_generator(seed_terms: list[str], min_len: int, max_len: int, shard_index: int, num_shards: int):
    for first_idx, first_term in enumerate(seed_terms):
        if first_idx % num_shards != shard_index:
            continue
        for length in range(min_len, max_len + 1):
            if length == 1:
                yield first_term, length, first_idx
                continue
            for suffix in product(seed_terms, repeat=length - 1):
                yield " ".join((first_term, *suffix)), length, first_idx


def batched(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def push_candidate(
    heap: list[tuple[float, int, dict]],
    keep_top_k: int,
    row: dict,
    tie_breaker,
) -> None:
    item = (float(row["mechanistic_score"]), next(tie_breaker), row)
    if len(heap) < keep_top_k:
        heapq.heappush(heap, item)
        return
    if item[0] <= heap[0][0]:
        return
    heapq.heapreplace(heap, item)


def score_phrase_batch(
    phrases: list[str],
    lengths: list[int],
    first_indices: list[int],
    *,
    config: SequenceMiningConfig,
    direction_unit: torch.Tensor,
    direction_meta: dict,
    tokenizer,
    model,
    baseline_activation: torch.Tensor,
    device: str,
) -> list[dict]:
    prompts = [build_injected_prompt(config.carrier_prompt, phrase) for phrase in phrases]
    activations = batch_terminal_activations(
        model=model,
        tokenizer=tokenizer,
        texts=prompts,
        layer_index=int(direction_meta["layer_index"]),
        device=device,
    )
    deltas = activations - baseline_activation.unsqueeze(0)
    scores = torch.matmul(deltas, direction_unit)
    delta_norms = deltas.norm(dim=1)

    rows: list[dict] = []
    for phrase, length, first_idx, score, delta_norm in zip(
        phrases,
        lengths,
        first_indices,
        scores.tolist(),
        delta_norms.tolist(),
    ):
        rows.append(
            {
                "steering_sentence": phrase,
                "mechanistic_score": float(score),
                "activation_delta_norm": float(delta_norm),
                "length": int(length),
                "direction_name": config.direction_name,
                "direction_sign": float(config.direction_sign),
                "search_mode": config.search_mode,
                "shard_index": config.shard_index,
                "first_term_index": int(first_idx),
                "carrier_prompt": config.carrier_prompt,
            }
        )
    return rows


def log_progress(total_sequences: int, retained: list[tuple[float, int, dict]], prefix: str = "") -> None:
    current_best = max(retained, key=lambda item: item[0])[0] if retained else 0.0
    lead = f"{prefix} " if prefix else ""
    print(
        f"{lead}scored_sequences={total_sequences} retained={len(retained)} "
        f"current_best={current_best:.4f}"
    )


def main() -> None:
    config = parse_args()
    device = choose_device()
    print(f"pro-human sequence mining config: {asdict(config)}")
    print(f"detected device: {device}")

    direction_unit, direction_meta = load_direction_vector(
        config.direction_tensors,
        direction_name=config.direction_name,
    )
    direction_unit = direction_unit * float(config.direction_sign)
    direction_meta = dict(direction_meta)
    direction_meta["direction_sign"] = float(config.direction_sign)

    seed_terms = load_lines(config.lexicon_file)
    tokenizer, model = load_model_and_tokenizer(config.model_name, device)

    baseline_prompt = build_chat_prompt(config.carrier_prompt)
    baseline_activation = batch_terminal_activations(
        model=model,
        tokenizer=tokenizer,
        texts=[baseline_prompt],
        layer_index=int(direction_meta["layer_index"]),
        device=device,
    )[0]

    total_sequences = 0
    retained: list[tuple[float, int, dict]] = []
    tie_breaker = count()

    if config.search_mode == "exhaustive":
        for batch in batched(
            sequence_generator(
                seed_terms=seed_terms,
                min_len=config.min_phrase_len,
                max_len=config.max_phrase_len,
                shard_index=config.shard_index,
                num_shards=config.num_shards,
            ),
            config.batch_size,
        ):
            rows = score_phrase_batch(
                [item[0] for item in batch],
                [item[1] for item in batch],
                [item[2] for item in batch],
                config=config,
                direction_unit=direction_unit,
                direction_meta=direction_meta,
                tokenizer=tokenizer,
                model=model,
                baseline_activation=baseline_activation,
                device=device,
            )
            for row in rows:
                push_candidate(retained, config.retain_top_k, row, tie_breaker)
                total_sequences += 1

            if total_sequences % max(1, config.batch_size * 100) == 0:
                log_progress(total_sequences, retained)
    else:
        frontier_batch = [
            (term, 1, idx)
            for idx, term in enumerate(seed_terms)
            if idx % config.num_shards == config.shard_index
        ]
        frontier_rows: list[dict] = []
        for batch in batched(frontier_batch, config.batch_size):
            rows = score_phrase_batch(
                [item[0] for item in batch],
                [item[1] for item in batch],
                [item[2] for item in batch],
                config=config,
                direction_unit=direction_unit,
                direction_meta=direction_meta,
                tokenizer=tokenizer,
                model=model,
                baseline_activation=baseline_activation,
                device=device,
            )
            frontier_rows.extend(rows)
            for row in rows:
                push_candidate(retained, config.retain_top_k, row, tie_breaker)
                total_sequences += 1

        frontier_rows.sort(key=lambda row: row["mechanistic_score"], reverse=True)
        frontier_rows = frontier_rows[: config.beam_width]
        log_progress(total_sequences, retained, prefix="depth=1")

        for depth in range(2, config.max_phrase_len + 1):
            proposals: list[tuple[str, int, int]] = []
            for row in frontier_rows:
                parent_phrase = row["steering_sentence"]
                first_idx = int(row["first_term_index"])
                for term in seed_terms:
                    proposals.append((f"{parent_phrase} {term}", depth, first_idx))

            if not proposals:
                break

            next_frontier: list[dict] = []
            for batch in batched(proposals, config.batch_size):
                rows = score_phrase_batch(
                    [item[0] for item in batch],
                    [item[1] for item in batch],
                    [item[2] for item in batch],
                    config=config,
                    direction_unit=direction_unit,
                    direction_meta=direction_meta,
                    tokenizer=tokenizer,
                    model=model,
                    baseline_activation=baseline_activation,
                    device=device,
                )
                next_frontier.extend(rows)
                for row in rows:
                    push_candidate(retained, config.retain_top_k, row, tie_breaker)
                    total_sequences += 1

            next_frontier.sort(key=lambda row: row["mechanistic_score"], reverse=True)
            frontier_rows = next_frontier[: config.beam_width]
            log_progress(total_sequences, retained, prefix=f"depth={depth}")

    retained_rows = [row for _, _, row in sorted(retained, key=lambda item: item[0], reverse=True)]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "retained_candidates.jsonl", retained_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "direction": direction_meta,
                "num_seed_terms": len(seed_terms),
                "num_sequences_scored": total_sequences,
                "num_retained": len(retained_rows),
                "best_score": retained_rows[0]["mechanistic_score"] if retained_rows else None,
            },
            handle,
            indent=2,
        )

    print(f"saved retained candidates to {output_dir / 'retained_candidates.jsonl'}")
    print(f"saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
