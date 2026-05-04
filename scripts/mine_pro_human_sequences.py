from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from dataclasses import asdict, dataclass
from itertools import count, product
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.activations import (
    batch_mean_suffix_activation_from_token_ids,
    encode_text,
    load_model_and_tokenizer,
)
from humanity_direction.data import load_lines, write_jsonl
from humanity_direction.mechanistic_scoring import load_direction_vector
from humanity_direction.pairs import collect_prompts, load_pairs
from humanity_direction.prompting import build_chat_prompt, build_injected_prompt_parts
from humanity_direction.scoring import choose_device


DEFAULT_CARRIER_PROMPT = "Please continue with a helpful response."
DEFAULT_LEXICON_FILE = "data/lexicon.txt"
DEFAULT_PAIRS_PATH = "data/axes"
DEFAULT_PROBE_COMPLETION = "I will respond carefully and helpfully."


@dataclass(slots=True)
class SequenceMiningConfig:
    model_name: str
    direction_tensors: str
    direction_name: str
    direction_sign: float
    output_dir: str
    search_space: str = "lexicon"
    lexicon_file: str = DEFAULT_LEXICON_FILE
    prompts_file: str | None = None
    pairs_path: str | None = None
    axis: str | None = None
    axes: str | None = None
    carrier_prompt: str = DEFAULT_CARRIER_PROMPT
    probe_completion: str = DEFAULT_PROBE_COMPLETION
    search_mode: str = "exhaustive"
    min_phrase_len: int = 1
    max_phrase_len: int = 5
    batch_size: int = 128
    retain_top_k: int = 5000
    beam_width: int = 128
    expansion_alphabet_size: int = 256
    validation_fraction: float = 0.2
    split_seed: int = 0
    shard_index: int = 0
    num_shards: int = 1


@dataclass(slots=True)
class SearchUnit:
    index: int
    label: str
    decoded_text: str
    token_ids: tuple[int, ...]


@dataclass(slots=True)
class PromptSpec:
    prompt: str
    baseline_prefix_ids: list[int]
    injected_suffix_ids: list[int]


@dataclass(slots=True)
class CandidateSpec:
    unit_positions: tuple[int, ...]
    unit_sequence: list[str]
    steering_sentence: str
    token_ids: tuple[int, ...]
    first_unit_index: int
    length: int


def parse_args() -> SequenceMiningConfig:
    parser = argparse.ArgumentParser(
        description="Mine mechanistic steering sequences against a saved direction vector."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--direction-tensors", required=True)
    parser.add_argument("--direction-name", default="global")
    parser.add_argument("--direction-sign", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-space", choices=("lexicon", "tokenizer_vocab"), default="lexicon")
    parser.add_argument("--lexicon-file", default=DEFAULT_LEXICON_FILE)
    parser.add_argument("--prompts-file")
    parser.add_argument("--pairs-path")
    parser.add_argument("--axis")
    parser.add_argument("--axes", help="Comma-separated axis list to include when loading from --pairs-path.")
    parser.add_argument("--carrier-prompt", default=DEFAULT_CARRIER_PROMPT)
    parser.add_argument("--probe-completion", default=DEFAULT_PROBE_COMPLETION)
    parser.add_argument("--search-mode", choices=("exhaustive", "beam"), default="exhaustive")
    parser.add_argument("--min-phrase-len", type=int, default=1)
    parser.add_argument("--max-phrase-len", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retain-top-k", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--expansion-alphabet-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if args.min_phrase_len < 1:
        parser.error("--min-phrase-len must be >= 1")
    if args.max_phrase_len < args.min_phrase_len:
        parser.error("--max-phrase-len must be >= --min-phrase-len")
    if args.direction_sign == 0:
        parser.error("--direction-sign must be nonzero")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.beam_width < 1:
        parser.error("--beam-width must be >= 1")
    if args.expansion_alphabet_size < 1:
        parser.error("--expansion-alphabet-size must be >= 1")
    if args.validation_fraction < 0 or args.validation_fraction >= 1:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.axis and args.axes:
        parser.error("use only one of --axis or --axes")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")

    return SequenceMiningConfig(**vars(args))


def batched(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_suffix_activations_batched(
    token_sequences: list[list[int]],
    suffix_lengths: list[int],
    *,
    model,
    tokenizer,
    layer_index: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(token_sequences), batch_size):
        end = start + batch_size
        chunks.append(
            batch_mean_suffix_activation_from_token_ids(
                model=model,
                tokenizer=tokenizer,
                token_id_sequences=token_sequences[start:end],
                suffix_lengths=suffix_lengths[start:end],
                layer_index=layer_index,
                device=device,
            )
        )
    if not chunks:
        raise ValueError("cannot compute activations for an empty token batch")
    return torch.cat(chunks, dim=0)


def load_prompt_pool(config: SequenceMiningConfig) -> tuple[list[str], str]:
    if config.prompts_file:
        prompts = load_lines(config.prompts_file)
        source = "prompts_file"
    elif config.pairs_path:
        rows = load_pairs(config.pairs_path)
        if config.axis:
            rows = [row for row in rows if row.get("axis") == config.axis]
        elif config.axes:
            allowed_axes = {item.strip() for item in config.axes.split(",") if item.strip()}
            rows = [row for row in rows if row.get("axis") in allowed_axes]
        prompts = collect_prompts(rows)
        source = "pairs_path"
    else:
        prompts = [config.carrier_prompt]
        source = "carrier_prompt"

    if not prompts:
        raise ValueError("no prompts available for mechanistic scoring")
    return prompts, source


def split_prompt_pool(prompts: list[str], validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if len(prompts) < 2 or validation_fraction <= 0.0:
        return prompts, []

    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_fraction))
    validation_count = max(1, min(len(shuffled) - 1, validation_count))
    validation_prompts = shuffled[:validation_count]
    train_prompts = shuffled[validation_count:]
    return train_prompts, validation_prompts


def build_prompt_specs(prompts: list[str], tokenizer) -> tuple[list[int], list[PromptSpec]]:
    if not prompts:
        raise ValueError("cannot build prompt specs for an empty prompt pool")

    injected_prefix_text, _ = build_injected_prompt_parts(prompts[0])
    injected_prefix_ids = encode_text(tokenizer, injected_prefix_text)

    prompt_specs: list[PromptSpec] = []
    for prompt in prompts:
        baseline_prefix_ids = encode_text(tokenizer, build_chat_prompt(prompt))
        _, injected_suffix_text = build_injected_prompt_parts(prompt)
        injected_suffix_ids = encode_text(tokenizer, injected_suffix_text)
        prompt_specs.append(
            PromptSpec(
                prompt=prompt,
                baseline_prefix_ids=baseline_prefix_ids,
                injected_suffix_ids=injected_suffix_ids,
            )
        )
    return injected_prefix_ids, prompt_specs


def load_search_units(config: SequenceMiningConfig, tokenizer) -> list[SearchUnit]:
    if config.search_space == "lexicon":
        entries = load_lines(config.lexicon_file)
        units = [
            SearchUnit(
                index=index,
                label=entry,
                decoded_text=entry,
                token_ids=tuple(encode_text(tokenizer, entry)),
            )
            for index, entry in enumerate(entries)
            if entry.strip()
        ]
        return [unit for unit in units if unit.token_ids]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab = tokenizer.get_vocab()
    by_id = {token_id: token for token, token_id in vocab.items()}
    units: list[SearchUnit] = []
    for token_id in sorted(by_id):
        if token_id in special_ids:
            continue
        decoded_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_text == "" or not decoded_text.strip():
            continue
        units.append(
            SearchUnit(
                index=token_id,
                label=tokenizer.convert_ids_to_tokens(token_id),
                decoded_text=decoded_text,
                token_ids=(token_id,),
            )
        )
    return units


def make_candidate(
    unit_positions: tuple[int, ...],
    *,
    units: list[SearchUnit],
    tokenizer,
    search_space: str,
) -> CandidateSpec | None:
    chosen_units = [units[position] for position in unit_positions]
    first_unit_index = chosen_units[0].index

    if search_space == "lexicon":
        steering_sentence = " ".join(unit.label for unit in chosen_units)
        token_ids = tuple(encode_text(tokenizer, steering_sentence))
        unit_sequence = [unit.label for unit in chosen_units]
    else:
        token_ids = tuple(token_id for unit in chosen_units for token_id in unit.token_ids)
        steering_sentence = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        unit_sequence = [unit.label for unit in chosen_units]

    if not token_ids:
        return None

    return CandidateSpec(
        unit_positions=unit_positions,
        unit_sequence=unit_sequence,
        steering_sentence=steering_sentence,
        token_ids=token_ids,
        first_unit_index=first_unit_index,
        length=len(unit_positions),
    )


def candidate_key(candidate: CandidateSpec) -> str:
    return ",".join(str(token_id) for token_id in candidate.token_ids)


def build_full_sequences(
    candidates: list[CandidateSpec],
    prompt_specs: list[PromptSpec],
    injected_prefix_ids: list[int],
    probe_completion_ids: list[int],
) -> list[list[int]]:
    sequences: list[list[int]] = []
    for candidate in candidates:
        candidate_token_ids = list(candidate.token_ids)
        for prompt_spec in prompt_specs:
            prefix_ids = injected_prefix_ids + candidate_token_ids + prompt_spec.injected_suffix_ids
            sequences.append(prefix_ids + probe_completion_ids)
    return sequences


def compute_baseline_activations(
    prompt_specs: list[PromptSpec],
    probe_completion_ids: list[int],
    *,
    model,
    tokenizer,
    layer_index: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    suffix_lengths = [len(probe_completion_ids)] * len(prompt_specs)
    sequences = [
        prompt_spec.baseline_prefix_ids + probe_completion_ids
        for prompt_spec in prompt_specs
    ]
    return compute_suffix_activations_batched(
        model=model,
        tokenizer=tokenizer,
        token_sequences=sequences,
        suffix_lengths=suffix_lengths,
        layer_index=layer_index,
        device=device,
        batch_size=batch_size,
    )


def push_candidate(
    heap: list[tuple[float, int, dict]],
    keep_top_k: int,
    row: dict,
    tie_breaker,
) -> None:
    item = (float(row["selection_score"]), next(tie_breaker), row)
    if len(heap) < keep_top_k:
        heapq.heappush(heap, item)
        return
    if item[0] <= heap[0][0]:
        return
    heapq.heapreplace(heap, item)


def score_candidates_batch(
    candidates: list[CandidateSpec],
    *,
    config: SequenceMiningConfig,
    prompt_specs: list[PromptSpec],
    injected_prefix_ids: list[int],
    probe_completion_ids: list[int],
    baseline_activations: torch.Tensor,
    direction_unit: torch.Tensor,
    direction_meta: dict,
    tokenizer,
    model,
    device: str,
    score_field: str = "mechanistic_score",
    norm_field: str = "activation_delta_norm",
) -> list[dict]:
    prompt_count = len(prompt_specs)
    suffix_lengths = [len(probe_completion_ids)] * (len(candidates) * prompt_count)
    token_sequences = build_full_sequences(
        candidates,
        prompt_specs,
        injected_prefix_ids,
        probe_completion_ids,
    )
    activations = compute_suffix_activations_batched(
        model=model,
        tokenizer=tokenizer,
        token_sequences=token_sequences,
        suffix_lengths=suffix_lengths,
        layer_index=int(direction_meta["layer_index"]),
        device=device,
        batch_size=config.batch_size,
    )
    activations = activations.reshape(len(candidates), prompt_count, -1)
    deltas = activations - baseline_activations.unsqueeze(0)
    scores = torch.matmul(deltas, direction_unit).mean(dim=1)
    delta_norms = deltas.norm(dim=2).mean(dim=1)

    rows: list[dict] = []
    for candidate, score, delta_norm in zip(candidates, scores.tolist(), delta_norms.tolist()):
        row = {
            "candidate_key": candidate_key(candidate),
            "steering_sentence": candidate.steering_sentence,
            "unit_sequence": candidate.unit_sequence,
            "token_ids": list(candidate.token_ids),
            "token_length": len(candidate.token_ids),
            "length": candidate.length,
            score_field: float(score),
            norm_field: float(delta_norm),
            "selection_score": float(score) if score_field == "mechanistic_score" else None,
            "direction_name": config.direction_name,
            "direction_sign": float(config.direction_sign),
            "search_mode": config.search_mode,
            "search_space": config.search_space,
            "probe_completion": config.probe_completion,
            "shard_index": config.shard_index,
            "first_unit_index": int(candidate.first_unit_index),
            "num_prompts_scored": prompt_count,
        }
        rows.append(row)
    return rows


def log_progress(total_sequences: int, retained: list[tuple[float, int, dict]], prefix: str = "") -> None:
    current_best = max(retained, key=lambda item: item[0])[0] if retained else 0.0
    lead = f"{prefix} " if prefix else ""
    print(
        f"{lead}scored_sequences={total_sequences} retained={len(retained)} "
        f"current_best={current_best:.4f}"
    )


def score_validation_rows(
    rows: list[dict],
    *,
    config: SequenceMiningConfig,
    prompt_specs: list[PromptSpec],
    injected_prefix_ids: list[int],
    probe_completion_ids: list[int],
    baseline_activations: torch.Tensor,
    direction_unit: torch.Tensor,
    direction_meta: dict,
    tokenizer,
    model,
    device: str,
) -> None:
    if not prompt_specs or not rows:
        for row in rows:
            row["selection_score"] = float(row["mechanistic_score"])
        return

    for batch_rows in batched(rows, config.batch_size):
        candidates = [
            CandidateSpec(
                unit_positions=(),
                unit_sequence=list(row.get("unit_sequence", [])),
                steering_sentence=row["steering_sentence"],
                token_ids=tuple(int(token_id) for token_id in row["token_ids"]),
                first_unit_index=int(row["first_unit_index"]),
                length=int(row["length"]),
            )
            for row in batch_rows
        ]
        scored_rows = score_candidates_batch(
            candidates,
            config=config,
            prompt_specs=prompt_specs,
            injected_prefix_ids=injected_prefix_ids,
            probe_completion_ids=probe_completion_ids,
            baseline_activations=baseline_activations,
            direction_unit=direction_unit,
            direction_meta=direction_meta,
            tokenizer=tokenizer,
            model=model,
            device=device,
            score_field="validation_mechanistic_score",
            norm_field="validation_activation_delta_norm",
        )
        for row, validation_row in zip(batch_rows, scored_rows):
            row["validation_mechanistic_score"] = validation_row["validation_mechanistic_score"]
            row["validation_activation_delta_norm"] = validation_row["validation_activation_delta_norm"]
            row["selection_score"] = validation_row["validation_mechanistic_score"]


def dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


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

    prompts, prompt_source_kind = load_prompt_pool(config)
    train_prompts, validation_prompts = split_prompt_pool(
        prompts,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )

    tokenizer, model = load_model_and_tokenizer(config.model_name, device)
    probe_completion_ids = encode_text(tokenizer, config.probe_completion)
    if not probe_completion_ids:
        raise ValueError("probe completion must produce at least one token")

    train_injected_prefix_ids, train_prompt_specs = build_prompt_specs(train_prompts, tokenizer)
    validation_injected_prefix_ids, validation_prompt_specs = build_prompt_specs(validation_prompts, tokenizer) if validation_prompts else ([], [])
    train_baseline_activations = compute_baseline_activations(
        train_prompt_specs,
        probe_completion_ids,
        model=model,
        tokenizer=tokenizer,
        layer_index=int(direction_meta["layer_index"]),
        device=device,
        batch_size=config.batch_size,
    )
    validation_baseline_activations = (
        compute_baseline_activations(
            validation_prompt_specs,
            probe_completion_ids,
            model=model,
            tokenizer=tokenizer,
            layer_index=int(direction_meta["layer_index"]),
            device=device,
            batch_size=config.batch_size,
        )
        if validation_prompt_specs
        else None
    )

    units = load_search_units(config, tokenizer)
    if not units:
        raise ValueError("search space produced zero candidate units")

    total_sequences = 0
    retained: list[tuple[float, int, dict]] = []
    tie_breaker = count()

    if config.search_mode == "exhaustive":
        def exhaustive_positions():
            for position, unit in enumerate(units):
                if unit.index % config.num_shards != config.shard_index:
                    continue
                for length in range(config.min_phrase_len, config.max_phrase_len + 1):
                    if length == 1:
                        yield (position,)
                        continue
                    for suffix in product(range(len(units)), repeat=length - 1):
                        yield (position, *suffix)

        for batch_positions in batched(exhaustive_positions(), config.batch_size):
            candidates = [
                candidate
                for candidate in (
                    make_candidate(tuple(unit_positions), units=units, tokenizer=tokenizer, search_space=config.search_space)
                    for unit_positions in batch_positions
                )
                if candidate is not None
            ]
            if not candidates:
                continue
            rows = score_candidates_batch(
                candidates,
                config=config,
                prompt_specs=train_prompt_specs,
                injected_prefix_ids=train_injected_prefix_ids,
                probe_completion_ids=probe_completion_ids,
                baseline_activations=train_baseline_activations,
                direction_unit=direction_unit,
                direction_meta=direction_meta,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            for row in rows:
                push_candidate(retained, config.retain_top_k, row, tie_breaker)
                total_sequences += 1

            if total_sequences % max(1, config.batch_size * 100) == 0:
                log_progress(total_sequences, retained)
    else:
        seed_candidates = [
            candidate
            for candidate in (
                make_candidate((position,), units=units, tokenizer=tokenizer, search_space=config.search_space)
                for position, unit in enumerate(units)
                if unit.index % config.num_shards == config.shard_index
            )
            if candidate is not None
        ]

        frontier_pairs: list[tuple[CandidateSpec, dict]] = []
        for batch_candidates in batched(seed_candidates, config.batch_size):
            rows = score_candidates_batch(
                batch_candidates,
                config=config,
                prompt_specs=train_prompt_specs,
                injected_prefix_ids=train_injected_prefix_ids,
                probe_completion_ids=probe_completion_ids,
                baseline_activations=train_baseline_activations,
                direction_unit=direction_unit,
                direction_meta=direction_meta,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            frontier_pairs.extend(zip(batch_candidates, rows))
            for row in rows:
                if row["length"] >= config.min_phrase_len:
                    push_candidate(retained, config.retain_top_k, row, tie_breaker)
                total_sequences += 1

        frontier_pairs.sort(key=lambda item: item[1]["mechanistic_score"], reverse=True)
        frontier_candidates = [candidate for candidate, _ in frontier_pairs[: config.beam_width]]

        if config.search_space == "tokenizer_vocab":
            expansion_positions = dedupe_preserve_order(
                [candidate.unit_positions[0] for candidate, _ in frontier_pairs[: config.expansion_alphabet_size]]
            )
        else:
            expansion_positions = list(range(len(units)))

        log_progress(total_sequences, retained, prefix="depth=1")

        for depth in range(2, config.max_phrase_len + 1):
            proposals: list[CandidateSpec] = []
            for candidate in frontier_candidates:
                for position in expansion_positions:
                    proposal = make_candidate(
                        candidate.unit_positions + (position,),
                        units=units,
                        tokenizer=tokenizer,
                        search_space=config.search_space,
                    )
                    if proposal is not None:
                        proposals.append(proposal)

            if not proposals:
                break

            next_frontier_pairs: list[tuple[CandidateSpec, dict]] = []
            for batch_candidates in batched(proposals, config.batch_size):
                rows = score_candidates_batch(
                    batch_candidates,
                    config=config,
                    prompt_specs=train_prompt_specs,
                    injected_prefix_ids=train_injected_prefix_ids,
                    probe_completion_ids=probe_completion_ids,
                    baseline_activations=train_baseline_activations,
                    direction_unit=direction_unit,
                    direction_meta=direction_meta,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
                next_frontier_pairs.extend(zip(batch_candidates, rows))
                for row in rows:
                    if row["length"] >= config.min_phrase_len:
                        push_candidate(retained, config.retain_top_k, row, tie_breaker)
                    total_sequences += 1

            next_frontier_pairs.sort(key=lambda item: item[1]["mechanistic_score"], reverse=True)
            frontier_candidates = [candidate for candidate, _ in next_frontier_pairs[: config.beam_width]]
            log_progress(total_sequences, retained, prefix=f"depth={depth}")

    retained_rows = [
        row
        for _, _, row in sorted(retained, key=lambda item: item[0], reverse=True)
        if row["length"] >= config.min_phrase_len
    ]

    score_validation_rows(
        retained_rows,
        config=config,
        prompt_specs=validation_prompt_specs,
        injected_prefix_ids=validation_injected_prefix_ids,
        probe_completion_ids=probe_completion_ids,
        baseline_activations=validation_baseline_activations if validation_baseline_activations is not None else torch.empty(0),
        direction_unit=direction_unit,
        direction_meta=direction_meta,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    retained_rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "retained_candidates.jsonl", retained_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "direction": direction_meta,
                "prompt_source": {
                    "kind": prompt_source_kind,
                    "prompts_file": config.prompts_file,
                    "pairs_path": config.pairs_path,
                    "axis": config.axis,
                    "axes": config.axes,
                },
                "num_total_prompts": len(prompts),
                "num_train_prompts": len(train_prompt_specs),
                "num_validation_prompts": len(validation_prompt_specs),
                "probe_completion": config.probe_completion,
                "num_search_units": len(units),
                "num_sequences_scored": total_sequences,
                "num_retained": len(retained_rows),
                "best_train_score": retained_rows[0]["mechanistic_score"] if retained_rows else None,
                "best_selection_score": retained_rows[0]["selection_score"] if retained_rows else None,
            },
            handle,
            indent=2,
        )

    print(f"saved retained candidates to {output_dir / 'retained_candidates.jsonl'}")
    print(f"saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
