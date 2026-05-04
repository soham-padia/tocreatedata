from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.activations import load_model_and_tokenizer, mean_completion_activation
from humanity_direction.data import load_lines, write_jsonl
from humanity_direction.mechanistic_scoring import load_direction_vector, projection_score
from humanity_direction.pairs import collect_prompts, load_pairs
from humanity_direction.prompting import build_chat_prompt, build_injected_prompt
from humanity_direction.scoring import choose_device, generate_completion
from humanity_direction.search import CandidateResult, beam_search_phrases


@dataclass(slots=True)
class MechanisticMiningConfig:
    model_name: str
    direction_tensors: str
    direction_name: str
    lexicon_file: str
    output_dir: str
    prompts_file: str | None = None
    pairs_path: str | None = None
    axis: str | None = None
    axes: str | None = None
    beam_width: int = 16
    max_phrase_len: int = 3
    keep_top_k: int = 1000
    max_new_tokens: int = 96
    temperature: float = 0.0


def parse_args() -> MechanisticMiningConfig:
    parser = argparse.ArgumentParser(
        description="Mine top candidate text sequences by mechanistic movement along a saved direction vector."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--direction-tensors", required=True)
    parser.add_argument("--direction-name", default="global")
    parser.add_argument("--lexicon-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--pairs-path")
    parser.add_argument("--axis")
    parser.add_argument("--axes", help="Comma-separated axis list to include when loading from --pairs-path.")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-phrase-len", type=int, default=3)
    parser.add_argument("--keep-top-k", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not args.prompts_file and not args.pairs_path:
        parser.error("one of --prompts-file or --pairs-path is required")
    if args.axis and args.axes:
        parser.error("use only one of --axis or --axes")
    return MechanisticMiningConfig(**vars(args))


def load_prompt_rows(config: MechanisticMiningConfig) -> tuple[list[str], dict[str, str]]:
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
    return prompts, prompt_axis


def main() -> None:
    config = parse_args()
    device = choose_device()
    print(f"mechanistic mining config: {asdict(config)}")
    print(f"detected device: {device}")

    prompts, prompt_axis = load_prompt_rows(config)
    seed_terms = load_lines(config.lexicon_file)
    direction_unit, direction_meta = load_direction_vector(
        config.direction_tensors,
        direction_name=config.direction_name,
    )
    tokenizer, model = load_model_and_tokenizer(config.model_name, device)

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
        activation = mean_completion_activation(
            model=model,
            tokenizer=tokenizer,
            prefix_text=prompt_text,
            completion_text=completion,
            layer_index=int(direction_meta["layer_index"]),
            device=device,
        )
        baseline_by_prompt[prompt] = {
            "completion": completion,
            "activation": activation,
        }

    def score_phrase(phrase: str) -> float:
        projections: list[float] = []
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
            injected_activation = mean_completion_activation(
                model=model,
                tokenizer=tokenizer,
                prefix_text=injected_prompt,
                completion_text=injected_completion,
                layer_index=int(direction_meta["layer_index"]),
                device=device,
            )
            delta = injected_activation - baseline_by_prompt[prompt]["activation"]
            projections.append(projection_score(delta, direction_unit))
        return sum(projections) / max(1, len(projections))

    ranked: list[CandidateResult] = beam_search_phrases(
        seed_terms=seed_terms,
        scorer=score_phrase,
        beam_width=config.beam_width,
        max_phrase_len=config.max_phrase_len,
        min_improvement=0.0,
        keep_top_k=config.keep_top_k,
        joiner=" ",
    )

    top_candidates: list[dict] = []
    dataset_rows: list[dict] = []
    for item in ranked:
        per_prompt: list[dict] = []
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
            injected_activation = mean_completion_activation(
                model=model,
                tokenizer=tokenizer,
                prefix_text=injected_prompt,
                completion_text=injected_completion,
                layer_index=int(direction_meta["layer_index"]),
                device=device,
            )
            baseline = baseline_by_prompt[prompt]
            delta = injected_activation - baseline["activation"]
            mechanistic_score = projection_score(delta, direction_unit)
            row = {
                "prompt": prompt,
                "prompt_axis": prompt_axis.get(prompt),
                "steering_sentence": item.phrase,
                "baseline_completion": baseline["completion"],
                "completion": injected_completion,
                "mechanistic_score": mechanistic_score,
                "direction_name": config.direction_name,
            }
            per_prompt.append(row)
            dataset_rows.append(row)

        top_candidates.append(
            {
                "steering_sentence": item.phrase,
                "score": item.score,
                "depth": item.depth,
                "direction_name": config.direction_name,
                "per_prompt": per_prompt,
            }
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "top_sentences.jsonl", top_candidates)
    write_jsonl(output_dir / "dataset.jsonl", dataset_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "direction": direction_meta,
                "num_prompts": len(prompts),
                "num_top_sentences": len(top_candidates),
                "num_dataset_rows": len(dataset_rows),
            },
            handle,
            indent=2,
        )

    print(f"saved top sentences to {output_dir / 'top_sentences.jsonl'}")
    print(f"saved mechanistic dataset to {output_dir / 'dataset.jsonl'}")
    print(f"saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
