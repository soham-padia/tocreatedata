from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.data import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge retained mechanistic candidate shards into one final top-K set.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--final-top-k", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    shard_files = sorted(input_root.glob("shards/shard_*/retained_candidates.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"no retained shard files found under {input_root}")

    best_by_phrase: dict[str, dict] = {}
    shard_summaries: list[dict] = []
    for file_path in shard_files:
        rows = load_jsonl(file_path)
        for row in rows:
            key = row.get("candidate_key", row["steering_sentence"])
            current = best_by_phrase.get(key)
            row_selection = row.get("selection_score", row["mechanistic_score"])
            current_selection = None if current is None else current.get("selection_score", current["mechanistic_score"])
            if current is None or row_selection > current_selection:
                best_by_phrase[key] = row
        summary_path = file_path.parent / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                shard_summaries.append(json.load(handle))

    final_rows = sorted(
        best_by_phrase.values(),
        key=lambda row: (
            row.get("selection_score", row["mechanistic_score"]),
            row["mechanistic_score"],
        ),
        reverse=True,
    )[: args.final_top_k]

    for idx, row in enumerate(final_rows, start=1):
        row["global_rank"] = idx

    write_jsonl(input_root / "top_sentences.jsonl", final_rows)
    with (input_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_root": str(input_root),
                "num_shards": len(shard_files),
                "num_unique_candidates": len(best_by_phrase),
                "final_top_k": args.final_top_k,
                "best_score": final_rows[0]["mechanistic_score"] if final_rows else None,
                "best_selection_score": final_rows[0].get("selection_score") if final_rows else None,
                "shards": shard_summaries,
            },
            handle,
            indent=2,
        )

    print(f"saved merged top sentences to {input_root / 'top_sentences.jsonl'}")
    print(f"saved merged summary to {input_root / 'summary.json'}")


if __name__ == "__main__":
    main()
