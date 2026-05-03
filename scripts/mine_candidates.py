from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.config import MiningConfig
from humanity_direction.data import load_lines, write_jsonl
from humanity_direction.direction import load_direction_spec, score_text_against_direction
from humanity_direction.pairs import collect_prompts, load_pairs
from humanity_direction.prompting import build_chat_prompt, build_injected_prompt
from humanity_direction.scoring import choose_device, generate_completion
from humanity_direction.search import CandidateResult, beam_search_phrases


def write_axis_split_outputs(
    candidate_rows: list[dict],
    dataset_rows: list[dict],
    output_file: str,
    dataset_file: str,
) -> None:
    output_root = Path(output_file).parent / "by_axis"
    dataset_root = Path(dataset_file).parent / "by_axis"
    axes = sorted({row["prompt_axis"] for row in dataset_rows})

    for axis_name in axes:
        axis_dataset_rows = [row for row in dataset_rows if row["prompt_axis"] == axis_name]
        axis_candidate_rows: list[dict] = []
        for row in candidate_rows:
            axis_per_prompt = [item for item in row["per_prompt"] if item["prompt_axis"] == axis_name]
            if not axis_per_prompt:
                continue
            axis_score = sum(item["score_delta"] for item in axis_per_prompt) / len(axis_per_prompt)
            axis_candidate_rows.append(
                {
                    "axis": axis_name,
                    "phrase": row["phrase"],
                    "score": axis_score,
                    "global_score": row["score"],
                    "depth": row["depth"],
                    "direction_name": row["direction_name"],
                    "per_prompt": axis_per_prompt,
                }
            )

        axis_candidate_rows.sort(key=lambda item: item["score"], reverse=True)
        write_jsonl(output_root / axis_name / "candidates.jsonl", axis_candidate_rows)
        write_jsonl(dataset_root / axis_name / "dataset.jsonl", axis_dataset_rows)


def parse_args() -> MiningConfig:
    parser = argparse.ArgumentParser(description="Mine phrases that move a base model toward a direction rubric.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--direction-file", required=True)
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--pairs-path")
    parser.add_argument("--axis")
    parser.add_argument("--axes", help="Comma-separated axis list to include when loading from --pairs-path.")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-phrase-len", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not args.prompts_file and not args.pairs_path:
        parser.error("one of --prompts-file or --pairs-path is required")
    if args.axis and args.axes:
        parser.error("use only one of --axis or --axes")
    return MiningConfig(**vars(args))


def load_model(model_name: str, device: str):
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


def main() -> None:
    config = parse_args()
    device = choose_device()
    print(f"mining config: {asdict(config)}")
    print(f"detected device: {device}")

    direction = load_direction_spec(config.direction_file)
    prompt_axis: dict[str, str] = {}
    if config.pairs_path:
        rows = load_pairs(config.pairs_path)
        if config.axis:
            rows = [row for row in rows if row.get("axis") == config.axis]
        elif config.axes:
            allowed_axes = {item.strip() for item in config.axes.split(",") if item.strip()}
            rows = [row for row in rows if row.get("axis") in allowed_axes]
        prompt_axis = {row["prompt"].strip(): row["axis"] for row in rows}
        prompts = collect_prompts(rows)
    else:
        prompts = load_lines(config.prompts_file)
    seed_terms = load_lines(config.lexicon_file)
    tokenizer, model = load_model(config.model_name, device)

    baseline_by_prompt: dict[str, dict] = {}
    for prompt in prompts:
        prompt_text = build_chat_prompt(prompt)
        completion = generate_completion(
            model,
            tokenizer,
            prompt_text,
            device=device,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        score = score_text_against_direction(completion, direction)
        baseline_by_prompt[prompt] = {"completion": completion, "score": score}

    def score_phrase(phrase: str) -> float:
        deltas: list[float] = []
        for prompt in prompts:
            injected_prompt = build_injected_prompt(prompt, phrase)
            injected_completion = generate_completion(
                model,
                tokenizer,
                injected_prompt,
                device=device,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            injected_score = score_text_against_direction(injected_completion, direction)
            baseline_score = baseline_by_prompt[prompt]["score"]
            deltas.append(injected_score.total - baseline_score.total)
        return sum(deltas) / max(1, len(deltas))

    ranked: list[CandidateResult] = beam_search_phrases(
        seed_terms=seed_terms,
        scorer=score_phrase,
        beam_width=config.beam_width,
        max_phrase_len=config.max_phrase_len,
        min_improvement=0.0,
    )
    top_ranked = ranked[: config.top_k]

    candidate_rows: list[dict] = []
    dataset_rows: list[dict] = []
    for item in top_ranked:
        per_prompt_scores = []
        for prompt in prompts:
            injected_prompt = build_injected_prompt(prompt, item.phrase)
            injected_completion = generate_completion(
                model,
                tokenizer,
                injected_prompt,
                device=device,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            injected_score = score_text_against_direction(injected_completion, direction)
            baseline = baseline_by_prompt[prompt]
            delta = injected_score.total - baseline["score"].total
            per_prompt_scores.append(
                {
                    "prompt": prompt,
                    "prompt_axis": prompt_axis.get(prompt),
                    "score_delta": delta,
                    "baseline_completion": baseline["completion"],
                    "baseline_score": baseline["score"].total,
                    "completion": injected_completion,
                    "direction_score": injected_score.total,
                    "breakdown": [
                        {
                            "axis": axis.axis,
                            "positive_hits": axis.positive_hits,
                            "negative_hits": axis.negative_hits,
                            "raw_score": axis.raw_score,
                            "weighted_score": axis.weighted_score,
                        }
                        for axis in injected_score.breakdown
                    ],
                }
            )
            dataset_rows.append(
                {
                    "prompt": prompt,
                    "prompt_axis": prompt_axis.get(prompt),
                    "steering_phrase": item.phrase,
                    "baseline_completion": baseline["completion"],
                    "baseline_score": baseline["score"].total,
                    "completion": injected_completion,
                    "direction_score": injected_score.total,
                    "score_delta": delta,
                    "direction_name": direction.name,
                }
            )

        candidate_rows.append(
            {
                "phrase": item.phrase,
                "score": item.score,
                "depth": item.depth,
                "direction_name": direction.name,
                "per_prompt": per_prompt_scores,
            }
        )

    write_jsonl(config.output_file, candidate_rows)
    write_jsonl(config.dataset_file, dataset_rows)
    if len({value for value in prompt_axis.values() if value}) > 1:
        write_axis_split_outputs(candidate_rows, dataset_rows, config.output_file, config.dataset_file)
    print(f"saved candidate ranking to {config.output_file}")
    print(f"saved mined dataset to {config.dataset_file}")


if __name__ == "__main__":
    main()
