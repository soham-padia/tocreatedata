from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.data import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paste-friendly Markdown report for completed or partial mechanistic runs."
    )
    parser.add_argument("--run-root", action="append", required=True, help="Run root to summarize.")
    parser.add_argument("--output-file", help="Optional markdown output path.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidate rows to include per run.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_path_for_shard(shard_dir: Path) -> Path | None:
    for name in ("summary.json", "summary.partial.json"):
        candidate = shard_dir / name
        if candidate.exists():
            return candidate
    return None


def rows_path_for_shard(shard_dir: Path) -> Path | None:
    for name in ("retained_candidates.jsonl", "retained_candidates.partial.jsonl"):
        candidate = shard_dir / name
        if candidate.exists():
            return candidate
    return None


def format_score(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def collect_run(run_root: Path) -> dict:
    final_summary = run_root / "summary.json"
    final_rows = run_root / "top_sentences.jsonl"
    shard_dirs = sorted(run_root.glob("shards/shard_*"))

    shard_summaries: list[dict] = []
    merged_rows: dict[str, dict] = {}

    for shard_dir in shard_dirs:
        summary_path = summary_path_for_shard(shard_dir)
        rows_path = rows_path_for_shard(shard_dir)
        summary = load_json(summary_path) if summary_path else {}
        rows = load_jsonl(rows_path) if rows_path else []

        shard_name = shard_dir.name
        phase_state = summary.get("phase_state", {})
        phase = phase_state.get("phase", "complete" if summary.get("complete") else "unknown")
        num_sequences = summary.get("num_sequences_scored", summary.get("num_sequences_scored", 0))
        best_score = summary.get("best_selection_score", summary.get("best_score"))
        shard_summaries.append(
            {
                "shard": shard_name,
                "phase": phase,
                "num_sequences_scored": num_sequences,
                "num_retained": summary.get("num_retained", len(rows)),
                "best_score": best_score,
                "summary_path": str(summary_path) if summary_path else None,
                "rows_path": str(rows_path) if rows_path else None,
            }
        )

        for row in rows:
            key = row.get("candidate_key", row["steering_sentence"])
            score = row.get("selection_score", row.get("mechanistic_score"))
            current = merged_rows.get(key)
            current_score = None if current is None else current.get("selection_score", current.get("mechanistic_score"))
            if current is None or float(score) > float(current_score):
                merged_rows[key] = row

    if final_summary.exists():
        final_summary_data = load_json(final_summary)
    else:
        final_summary_data = None

    if final_rows.exists():
        final_rows_data = load_jsonl(final_rows)
    else:
        final_rows_data = sorted(
            merged_rows.values(),
            key=lambda row: (
                float(row.get("selection_score", row.get("mechanistic_score", float("-inf")))),
                float(row.get("mechanistic_score", float("-inf"))),
            ),
            reverse=True,
        )

    config = None
    direction = None
    if final_summary_data and final_summary_data.get("shards"):
        first_shard = final_summary_data["shards"][0]
        config = first_shard.get("config")
        direction = first_shard.get("direction")
    elif shard_summaries:
        first_summary_path = shard_summaries[0]["summary_path"]
        if first_summary_path:
            first_summary = load_json(Path(first_summary_path))
            config = first_summary.get("config")
            direction = first_summary.get("direction")

    return {
        "run_root": str(run_root),
        "complete": final_summary.exists() and final_rows.exists(),
        "config": config or {},
        "direction": direction or {},
        "shards": shard_summaries,
        "rows": final_rows_data,
    }


def render_run(run: dict, top_k: int) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Run: `{run['run_root']}`")
    lines.append("")

    config = run["config"]
    direction = run["direction"]
    shard_phases = ", ".join(f"{shard['shard']}={shard['phase']}" for shard in run["shards"]) or "n/a"
    total_scored = sum(int(shard.get("num_sequences_scored") or 0) for shard in run["shards"])
    complete = "yes" if run["complete"] else "no"

    lines.append("### Overview")
    lines.append("")
    lines.append(f"- Complete: `{complete}`")
    lines.append(f"- Search space: `{config.get('search_space', 'n/a')}`")
    lines.append(f"- Search mode: `{config.get('search_mode', 'n/a')}`")
    lines.append(f"- Direction sign: `{config.get('direction_sign', 'n/a')}`")
    lines.append(f"- Min/Max phrase len: `{config.get('min_phrase_len', 'n/a')}` / `{config.get('max_phrase_len', 'n/a')}`")
    lines.append(f"- Beam width: `{config.get('beam_width', 'n/a')}`")
    lines.append(f"- Batch size: `{config.get('batch_size', 'n/a')}`")
    lines.append(f"- Validation fraction: `{config.get('validation_fraction', 'n/a')}`")
    lines.append(f"- Direction layer: `{direction.get('layer_index', 'n/a')}`")
    lines.append(f"- Direction norm: `{format_score(direction.get('vector_norm'))}`")
    lines.append(f"- Total sequences scored across shards: `{total_scored}`")
    lines.append(f"- Shard phases: `{shard_phases}`")
    lines.append("")

    lines.append("### Shards")
    lines.append("")
    lines.append("| Shard | Phase | Scored | Retained | Best score |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for shard in run["shards"]:
        lines.append(
            f"| `{shard['shard']}` | `{shard['phase']}` | `{shard['num_sequences_scored']}` | "
            f"`{shard['num_retained']}` | `{format_score(shard['best_score'])}` |"
        )
    lines.append("")

    lines.append(f"### Top {min(top_k, len(run['rows']))} Candidates")
    lines.append("")
    lines.append("| Rank | Text | Selection score | Train score | Token length |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for index, row in enumerate(run["rows"][:top_k], start=1):
        text = row["steering_sentence"].replace("\n", "\\n").replace("\r", "\\r")
        lines.append(
            f"| {index} | `{text}` | `{format_score(row.get('selection_score'))}` | "
            f"`{format_score(row.get('mechanistic_score'))}` | `{row.get('token_length', 'n/a')}` |"
        )
    lines.append("")

    return lines


def render_report(runs: list[dict], top_k: int) -> str:
    lines = [
        "# Mechanistic Progress Report",
        "",
        "This report summarizes the current state of the mechanistic mining runs, including partial checkpointed runs.",
        "",
    ]
    for run in runs:
        lines.extend(render_run(run, top_k=top_k))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = [collect_run(Path(run_root)) for run_root in args.run_root]
    report = render_report(runs, top_k=args.top_k)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"wrote {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
