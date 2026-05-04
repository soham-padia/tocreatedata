from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.activations import load_model_and_tokenizer
from humanity_direction.data import write_jsonl
from humanity_direction.scoring import choose_device


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_FILE = "data/token_shortlists/qwen25_diverse_5000.jsonl"


@dataclass(slots=True)
class ShortlistConfig:
    model_name: str
    output_file: str
    shortlist_size: int
    weird_reserve: int
    clean_pool_size: int
    weird_pool_size: int
    project_dim: int
    seed: int
    device: str | None = None


@dataclass(slots=True)
class TokenRecord:
    token_id: int
    token: str
    decoded_text: str
    token_ids: list[int]
    bucket: str
    weird_score: int
    flags: list[str]


def parse_args() -> ShortlistConfig:
    parser = argparse.ArgumentParser(
        description="Build a tokenizer-vocab shortlist using embedding-space diversity with a reserved weird-token bucket."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--shortlist-size", type=int, default=5000)
    parser.add_argument("--weird-reserve", type=int, default=500)
    parser.add_argument(
        "--clean-pool-size",
        type=int,
        default=20000,
        help="Random clean-token pool size before diversity selection.",
    )
    parser.add_argument(
        "--weird-pool-size",
        type=int,
        default=5000,
        help="High-weirdness candidate pool size before diversity selection.",
    )
    parser.add_argument(
        "--project-dim",
        type=int,
        default=32,
        help="Random projection dimension used for approximate diversity selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.shortlist_size < 1:
        parser.error("--shortlist-size must be >= 1")
    if args.weird_reserve < 0 or args.weird_reserve > args.shortlist_size:
        parser.error("--weird-reserve must be in [0, shortlist-size]")
    if args.clean_pool_size < 1:
        parser.error("--clean-pool-size must be >= 1")
    if args.weird_pool_size < 1:
        parser.error("--weird-pool-size must be >= 1")
    if args.project_dim < 2:
        parser.error("--project-dim must be >= 2")

    return ShortlistConfig(**vars(args))


def classify_token(token_id: int, token: str, decoded_text: str) -> TokenRecord | None:
    if decoded_text == "":
        return None

    if decoded_text.strip() == "":
        return None

    visible = "".join(char for char in decoded_text if not char.isspace())
    if visible == "":
        return None

    flags: list[str] = []
    weird_score = 0

    if any(char in "\n\r\t" or ord(char) < 32 for char in decoded_text):
        flags.append("control")
        weird_score += 5

    if "\ufffd" in decoded_text:
        flags.append("replacement")
        weird_score += 4

    if visible and all(not char.isalnum() for char in visible):
        flags.append("punct_only")
        weird_score += 4

    if any(char in "{}[]();:=<>/\\_*`" for char in visible):
        flags.append("code_punct")
        weird_score += 2

    if any(ord(char) > 126 for char in visible if char != "\ufffd"):
        flags.append("non_ascii")
        weird_score += 2

    fragment_text = visible
    if (
        fragment_text.isalpha()
        and not decoded_text.startswith(" ")
        and fragment_text.lower() == fragment_text
        and len(fragment_text) <= 4
    ):
        flags.append("fragment")
        weird_score += 2

    bucket = "weird" if flags else "clean"
    return TokenRecord(
        token_id=token_id,
        token=token,
        decoded_text=decoded_text,
        token_ids=[token_id],
        bucket=bucket,
        weird_score=weird_score,
        flags=flags,
    )


def collect_token_records(tokenizer) -> tuple[list[TokenRecord], list[TokenRecord], dict]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab = tokenizer.get_vocab()
    by_id = {token_id: token for token, token_id in vocab.items()}

    clean_records: list[TokenRecord] = []
    weird_records: list[TokenRecord] = []
    skipped = 0

    for token_id in sorted(by_id):
        if token_id in special_ids:
            skipped += 1
            continue

        decoded_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        record = classify_token(token_id, by_id[token_id], decoded_text)
        if record is None:
            skipped += 1
            continue
        if record.bucket == "weird":
            weird_records.append(record)
        else:
            clean_records.append(record)

    stats = {
        "raw_vocab_size": len(vocab),
        "special_or_dropped": skipped,
        "clean_candidates": len(clean_records),
        "weird_candidates": len(weird_records),
    }
    return clean_records, weird_records, stats


def random_sample_records(records: list[TokenRecord], limit: int, seed: int) -> list[TokenRecord]:
    if len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:limit]


def top_weird_records(records: list[TokenRecord], limit: int, seed: int) -> list[TokenRecord]:
    if len(records) <= limit:
        return list(records)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(
        key=lambda record: (
            record.weird_score,
            len(record.decoded_text.strip()),
        ),
        reverse=True,
    )
    return shuffled[:limit]


def select_diverse_records(
    records: list[TokenRecord],
    *,
    target_size: int,
    embedding_weight: torch.Tensor,
    device: str,
    project_dim: int,
    seed: int,
    log_prefix: str,
) -> list[TokenRecord]:
    if target_size <= 0 or not records:
        return []
    if len(records) <= target_size:
        return list(records)

    token_ids = torch.tensor([record.token_id for record in records], dtype=torch.long, device=embedding_weight.device)
    embeddings = embedding_weight.index_select(0, token_ids).float()
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)

    generator = torch.Generator(device=embedding_weight.device if embedding_weight.device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    projection = torch.randn(
        embeddings.shape[1],
        project_dim,
        generator=generator,
        device=embedding_weight.device,
        dtype=embeddings.dtype,
    )
    projected = embeddings @ projection
    projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-8)

    centroid = projected.mean(dim=0, keepdim=True)
    start_index = int(torch.argmax((projected - centroid).norm(dim=1)).item())

    min_distance = torch.full((projected.shape[0],), float("inf"), device=projected.device)
    selected_indices: list[int] = []
    current_index = start_index

    for step in range(target_size):
        selected_indices.append(current_index)
        similarity = projected @ projected[current_index]
        distance = 1.0 - similarity
        min_distance = torch.minimum(min_distance, distance)
        min_distance[current_index] = -1.0
        if step + 1 == target_size:
            break
        current_index = int(torch.argmax(min_distance).item())
        if (step + 1) % 500 == 0 or step + 1 == target_size:
            print(f"{log_prefix}: selected {step + 1}/{target_size}")

    return [records[index] for index in selected_indices]


def main() -> None:
    config = parse_args()
    device = config.device or choose_device()
    print(f"building token shortlist with config={asdict(config)}")
    print(f"detected device: {device}")

    tokenizer, model = load_model_and_tokenizer(config.model_name, device)
    embedding_weight = model.get_input_embeddings().weight.detach()

    clean_records, weird_records, stats = collect_token_records(tokenizer)
    print(f"candidate stats: {stats}")

    weird_target = min(config.weird_reserve, config.shortlist_size)
    clean_target = config.shortlist_size - weird_target

    clean_pool = random_sample_records(clean_records, config.clean_pool_size, seed=config.seed)
    weird_pool = top_weird_records(weird_records, config.weird_pool_size, seed=config.seed + 1)

    print(
        f"selected candidate pools: clean_pool={len(clean_pool)} weird_pool={len(weird_pool)} "
        f"clean_target={clean_target} weird_target={weird_target}"
    )

    selected_weird = select_diverse_records(
        weird_pool,
        target_size=min(weird_target, len(weird_pool)),
        embedding_weight=embedding_weight,
        device=device,
        project_dim=config.project_dim,
        seed=config.seed + 11,
        log_prefix="weird",
    )

    clean_target = config.shortlist_size - len(selected_weird)
    selected_clean = select_diverse_records(
        clean_pool,
        target_size=min(clean_target, len(clean_pool)),
        embedding_weight=embedding_weight,
        device=device,
        project_dim=config.project_dim,
        seed=config.seed + 23,
        log_prefix="clean",
    )

    shortlist = selected_clean + selected_weird
    if len(shortlist) < config.shortlist_size:
        clean_seen = {record.token_id for record in shortlist}
        clean_fill = [record for record in clean_pool if record.token_id not in clean_seen]
        needed = min(config.shortlist_size - len(shortlist), len(clean_fill))
        shortlist.extend(clean_fill[:needed])

    output_rows = [asdict(record) for record in shortlist]
    output_path = Path(config.output_file)
    write_jsonl(output_path, output_rows)

    summary = {
        "config": asdict(config),
        "device": device,
        "stats": stats,
        "selected_clean": sum(record["bucket"] == "clean" for record in output_rows),
        "selected_weird": sum(record["bucket"] == "weird" for record in output_rows),
        "final_size": len(output_rows),
        "sample_clean": output_rows[:10],
        "sample_weird": [row for row in output_rows if row["bucket"] == "weird"][:10],
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote shortlist to {output_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
