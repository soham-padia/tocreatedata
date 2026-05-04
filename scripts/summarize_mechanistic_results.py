from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunData:
    label: str
    root: Path
    rows: list[dict]
    summary: dict
    segmented_units: list[list[str]]


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce a detailed report for one or more mechanistic mining runs."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Run root containing top_sentences.jsonl and summary.json. Repeat for comparison.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional label for each --run-root. Must match the number of run roots if provided.",
    )
    parser.add_argument("--lexicon-file", default="data/lexicon.txt")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--top-overlap-k",
        default="10,25,50,100,250,500,1000",
        help="Comma-separated cutoffs used for overlap comparisons.",
    )
    parser.add_argument(
        "--top-show",
        type=int,
        default=15,
        help="How many top sequences to show in the markdown report.",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def numeric_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def top_slice_mean(values: list[float], top_n: int) -> float | None:
    if not values:
        return None
    slice_values = values[: min(top_n, len(values))]
    return float(statistics.fmean(slice_values))


def default_output_dir(run_roots: list[Path], labels: list[str] | None) -> Path:
    if len(run_roots) == 1:
        return run_roots[0] / "analysis"
    stem = "_vs_".join(labels or [path.name for path in run_roots[:3]])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / "mechanistic_reports" / f"{stem}_{timestamp}"


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(root: Path, label: str, lexicon: list[str]) -> RunData:
    rows = load_jsonl(root / "top_sentences.jsonl")
    summary = load_summary(root / "summary.json")
    segmented_units = [
        list(row["unit_sequence"])
        if isinstance(row.get("unit_sequence"), list) and row["unit_sequence"]
        else segment_sequence(row["steering_sentence"], lexicon, int(row["length"]))
        for row in rows
    ]
    return RunData(
        label=label,
        root=root,
        rows=rows,
        summary=summary,
        segmented_units=segmented_units,
    )


def segment_sequence(text: str, lexicon: list[str], target_len: int) -> list[str]:
    words = text.split()
    lexicon_tokens = [(entry, entry.split()) for entry in sorted(lexicon, key=lambda item: (-len(item.split()), item))]

    @lru_cache(maxsize=None)
    def dfs(word_index: int, remaining_units: int):
        if word_index == len(words) and remaining_units == 0:
            return ()
        if word_index >= len(words) or remaining_units <= 0:
            return None
        for entry, entry_tokens in lexicon_tokens:
            width = len(entry_tokens)
            if word_index + width > len(words):
                continue
            if words[word_index : word_index + width] != entry_tokens:
                continue
            result = dfs(word_index + width, remaining_units - 1)
            if result is not None:
                return (entry, *result)
        return None

    found = dfs(0, target_len)
    if found is None:
        return [text]
    return list(found)


def summarize_run(run: RunData) -> dict:
    rows = run.rows
    segmented = run.segmented_units
    scores = [float(row["mechanistic_score"]) for row in rows]
    selection_scores = [
        float(row.get("selection_score", row["mechanistic_score"]))
        for row in rows
    ]
    delta_norms = [float(row["activation_delta_norm"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    validation_scores = [
        float(row["validation_mechanistic_score"])
        for row in rows
        if row.get("validation_mechanistic_score") is not None
    ]

    unit_counter: Counter[str] = Counter()
    unit_score_totals: defaultdict[str, float] = defaultdict(float)
    start_counter: Counter[str] = Counter()
    end_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    trigram_counter: Counter[str] = Counter()
    unique_units_per_row: list[int] = []
    repeat_fraction_per_row: list[float] = []
    max_unit_fraction_per_row: list[float] = []
    segmentation_failures = 0

    length_buckets: defaultdict[int, list[dict]] = defaultdict(list)
    first_term_counter: Counter[str] = Counter()

    for row, units in zip(rows, segmented):
        score = float(row["mechanistic_score"])
        length_buckets[int(row["length"])].append(row)
        if len(units) == 1 and units[0] == row["steering_sentence"] and int(row["length"]) != 1:
            segmentation_failures += 1
        local_counter = Counter(units)
        unique_units_per_row.append(len(local_counter))
        repeat_fraction_per_row.append(1.0 - (len(local_counter) / max(1, len(units))))
        max_unit_fraction_per_row.append(max(local_counter.values()) / max(1, len(units)))
        if units:
            start_counter[units[0]] += 1
            end_counter[units[-1]] += 1
            first_term_counter[units[0]] += 1
        for unit in units:
            unit_counter[unit] += 1
            unit_score_totals[unit] += score
        for idx in range(len(units) - 1):
            bigram_counter[f"{units[idx]} -> {units[idx + 1]}"] += 1
        for idx in range(len(units) - 2):
            trigram_counter[f"{units[idx]} -> {units[idx + 1]} -> {units[idx + 2]}"] += 1

    per_length = {}
    for length, bucket_rows in sorted(length_buckets.items()):
        bucket_scores = [float(row["mechanistic_score"]) for row in bucket_rows]
        bucket_norms = [float(row["activation_delta_norm"]) for row in bucket_rows]
        per_length[str(length)] = {
            "count": len(bucket_rows),
            "score_mean": float(statistics.fmean(bucket_scores)),
            "score_max": float(max(bucket_scores)),
            "score_min": float(min(bucket_scores)),
            "delta_norm_mean": float(statistics.fmean(bucket_norms)),
        }

    top_examples = []
    for idx, (row, units) in enumerate(zip(rows[:15], segmented[:15]), start=1):
        top_examples.append(
            {
                "rank": idx,
                "steering_sentence": row["steering_sentence"],
                "segmented_units": units,
                "mechanistic_score": float(row["mechanistic_score"]),
                "activation_delta_norm": float(row["activation_delta_norm"]),
                "length": int(row["length"]),
            }
        )

    summary = {
        "label": run.label,
        "root": str(run.root),
        "num_rows": len(rows),
        "direction_name": rows[0]["direction_name"] if rows else None,
        "direction_sign": rows[0].get("direction_sign") if rows else None,
        "search_mode": rows[0].get("search_mode") if rows else None,
        "search_space": rows[0].get("search_space") if rows else None,
        "score_stats": numeric_stats(scores),
        "selection_score_stats": numeric_stats(selection_scores),
        "validation_score_stats": numeric_stats(validation_scores),
        "delta_norm_stats": numeric_stats(delta_norms),
        "length_stats": numeric_stats([float(length) for length in lengths]),
        "top_slice_score_means": {
            "top_1": top_slice_mean(scores, 1),
            "top_10": top_slice_mean(scores, 10),
            "top_50": top_slice_mean(scores, 50),
            "top_100": top_slice_mean(scores, 100),
            "top_250": top_slice_mean(scores, 250),
            "top_500": top_slice_mean(scores, 500),
            "top_1000": top_slice_mean(scores, 1000),
        },
        "per_length": per_length,
        "repetition": {
            "mean_unique_units": float(statistics.fmean(unique_units_per_row)) if unique_units_per_row else None,
            "mean_repeat_fraction": float(statistics.fmean(repeat_fraction_per_row)) if repeat_fraction_per_row else None,
            "mean_max_unit_fraction": float(statistics.fmean(max_unit_fraction_per_row)) if max_unit_fraction_per_row else None,
        },
        "segmentation_failures": segmentation_failures,
        "top_units": unit_counter.most_common(20),
        "top_units_by_score_mass": sorted(unit_score_totals.items(), key=lambda item: item[1], reverse=True)[:20],
        "top_start_units": start_counter.most_common(15),
        "top_end_units": end_counter.most_common(15),
        "top_bigrams": bigram_counter.most_common(20),
        "top_trigrams": trigram_counter.most_common(20),
        "first_term_distribution": first_term_counter.most_common(15),
        "top_examples": top_examples,
        "run_summary_file": run.summary,
    }
    return summary


def compare_runs(run_summaries: list[dict], top_cutoffs: list[int], lexicon: list[str]) -> dict:
    if len(run_summaries) < 2:
        return {}

    label_to_rows = {}
    for summary in run_summaries:
        rows = load_jsonl(Path(summary["root"]) / "top_sentences.jsonl")
        label_to_rows[summary["label"]] = rows

    overlap = []
    labels = list(label_to_rows.keys())
    if len(labels) == 2:
        left, right = labels
        left_rows = label_to_rows[left]
        right_rows = label_to_rows[right]
        for cutoff in top_cutoffs:
            left_set = {row["steering_sentence"] for row in left_rows[:cutoff]}
            right_set = {row["steering_sentence"] for row in right_rows[:cutoff]}
            intersection = left_set & right_set
            union = left_set | right_set
            overlap.append(
                {
                    "cutoff": cutoff,
                    "shared": len(intersection),
                    "jaccard": (len(intersection) / len(union)) if union else 0.0,
                }
            )

        left_unit_counts = Counter()
        right_unit_counts = Counter()
        left_segmented = load_run(Path(run_summaries[0]["root"]), left, lexicon).segmented_units
        right_segmented = load_run(Path(run_summaries[1]["root"]), right, lexicon).segmented_units
        for units in left_segmented:
            left_unit_counts.update(units)
        for units in right_segmented:
            right_unit_counts.update(units)

        unit_diff = []
        for unit in sorted(set(left_unit_counts) | set(right_unit_counts)):
            unit_diff.append(
                {
                    "unit": unit,
                    left: left_unit_counts[unit],
                    right: right_unit_counts[unit],
                    "difference": left_unit_counts[unit] - right_unit_counts[unit],
                }
            )
        unit_diff.sort(key=lambda row: abs(row["difference"]), reverse=True)

        return {
            "labels": labels,
            "overlap": overlap,
            "most_distinctive_units": unit_diff[:25],
        }

    return {"labels": labels}


def render_markdown(run_summaries: list[dict], comparison: dict, top_show: int) -> str:
    lines: list[str] = []
    lines.append("# Mechanistic Results Summary")
    lines.append("")
    for summary in run_summaries:
        lines.append(f"## {summary['label']}")
        lines.append("")
        lines.append(f"- Root: `{summary['root']}`")
        lines.append(f"- Direction: `{summary['direction_name']}`")
        lines.append(f"- Direction sign: `{summary['direction_sign']}`")
        lines.append(f"- Search mode: `{summary['search_mode']}`")
        lines.append(f"- Search space: `{summary['search_space']}`")
        lines.append(f"- Rows: `{summary['num_rows']}`")
        lines.append(f"- Segmentation failures: `{summary['segmentation_failures']}`")
        lines.append("")
        lines.append("### Score Stats")
        score_stats = summary["score_stats"]
        lines.append(
            f"- mean={score_stats['mean']:.4f} median={score_stats['median']:.4f} "
            f"min={score_stats['min']:.4f} max={score_stats['max']:.4f} stdev={score_stats['stdev']:.4f}"
        )
        lines.append(
            f"- p10={score_stats['p10']:.4f} p25={score_stats['p25']:.4f} "
            f"p75={score_stats['p75']:.4f} p90={score_stats['p90']:.4f}"
        )
        lines.append("")
        if summary["validation_score_stats"]:
            validation_stats = summary["validation_score_stats"]
            lines.append("### Validation Score Stats")
            lines.append(
                f"- mean={validation_stats['mean']:.4f} median={validation_stats['median']:.4f} "
                f"min={validation_stats['min']:.4f} max={validation_stats['max']:.4f} "
                f"stdev={validation_stats['stdev']:.4f}"
            )
            lines.append("")
        lines.append("### Delta-Norm Stats")
        delta_stats = summary["delta_norm_stats"]
        lines.append(
            f"- mean={delta_stats['mean']:.4f} median={delta_stats['median']:.4f} "
            f"min={delta_stats['min']:.4f} max={delta_stats['max']:.4f} stdev={delta_stats['stdev']:.4f}"
        )
        lines.append("")
        lines.append("### Top-Slice Means")
        for key, value in summary["top_slice_score_means"].items():
            lines.append(f"- {key}: {value:.4f}")
        lines.append("")
        lines.append("### Length Profile")
        for length, bucket in summary["per_length"].items():
            lines.append(
                f"- length {length}: count={bucket['count']} mean_score={bucket['score_mean']:.4f} "
                f"mean_delta_norm={bucket['delta_norm_mean']:.4f}"
            )
        lines.append("")
        lines.append("### Repetition")
        repetition = summary["repetition"]
        lines.append(
            f"- mean unique units per row: {repetition['mean_unique_units']:.3f}"
        )
        lines.append(
            f"- mean repeat fraction: {repetition['mean_repeat_fraction']:.3f}"
        )
        lines.append(
            f"- mean max-unit fraction: {repetition['mean_max_unit_fraction']:.3f}"
        )
        lines.append("")
        lines.append("### Top Units")
        for unit, count in summary["top_units"][:15]:
            lines.append(f"- {unit}: {count}")
        lines.append("")
        lines.append("### Top Bigrams")
        for unit, count in summary["top_bigrams"][:10]:
            lines.append(f"- {unit}: {count}")
        lines.append("")
        lines.append("### Top Trigrams")
        for unit, count in summary["top_trigrams"][:10]:
            lines.append(f"- {unit}: {count}")
        lines.append("")
        lines.append("### Top Examples")
        for example in summary["top_examples"][:top_show]:
            lines.append(
                f"- rank {example['rank']}: score={example['mechanistic_score']:.4f} "
                f"len={example['length']} units={example['segmented_units']} "
                f"text=`{example['steering_sentence']}`"
            )
        lines.append("")

    if comparison:
        lines.append("## Comparison")
        lines.append("")
        if "overlap" in comparison:
            lines.append("### Exact Overlap")
            for row in comparison["overlap"]:
                lines.append(
                    f"- top-{row['cutoff']}: shared={row['shared']} jaccard={row['jaccard']:.4f}"
                )
            lines.append("")
        if "most_distinctive_units" in comparison:
            lines.append("### Most Distinctive Units")
            for row in comparison["most_distinctive_units"][:15]:
                labels = comparison["labels"]
                lines.append(
                    f"- {row['unit']}: {labels[0]}={row[labels[0]]}, {labels[1]}={row[labels[1]]}, diff={row['difference']}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_roots = [Path(item) for item in args.run_root]
    labels = args.label
    if labels and len(labels) != len(run_roots):
        raise ValueError("number of --label values must match number of --run-root values")
    lexicon = load_lines(args.lexicon_file)
    labels = labels or [path.name for path in run_roots]
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(run_roots, labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    for root, label in zip(run_roots, labels):
        run = load_run(root, label, lexicon)
        run_summaries.append(summarize_run(run))

    top_cutoffs = [int(item.strip()) for item in args.top_overlap_k.split(",") if item.strip()]
    comparison = compare_runs(run_summaries, top_cutoffs, lexicon)

    report = {
        "generated_at": datetime.now().isoformat(),
        "run_summaries": run_summaries,
        "comparison": comparison,
    }

    markdown = render_markdown(run_summaries, comparison, args.top_show)

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(markdown)
    print(f"saved report json to {output_dir / 'report.json'}")
    print(f"saved report markdown to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
