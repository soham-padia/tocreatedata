from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.activations import load_model_and_tokenizer, mean_completion_activation
from humanity_direction.pairs import load_pairs
from humanity_direction.prompting import build_training_example
from humanity_direction.scoring import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activation-space direction vectors from chosen-vs-rejected axis pairs."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--axis")
    parser.add_argument("--axes", help="Comma-separated axis list to include.")
    parser.add_argument("--layer-index", type=int, default=-1)
    args = parser.parse_args()
    if args.axis and args.axes:
        parser.error("use only one of --axis or --axes")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device()
    print(f"extracting mechanistic directions with model={args.model_name} device={device} layer={args.layer_index}")

    rows = load_pairs(args.pairs_path)
    if args.axis:
        rows = [row for row in rows if row.get("axis") == args.axis]
    elif args.axes:
        allowed_axes = {item.strip() for item in args.axes.split(",") if item.strip()}
        rows = [row for row in rows if row.get("axis") in allowed_axes]

    if not rows:
        raise ValueError("no rows matched the requested axis filter")

    rows_by_axis: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_axis[row["axis"]].append(row)

    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axis_vectors: dict[str, torch.Tensor] = {}
    summary_rows: list[dict] = []
    all_diffs: list[torch.Tensor] = []

    for axis_name, axis_rows in sorted(rows_by_axis.items()):
        diffs: list[torch.Tensor] = []
        for row in axis_rows:
            prefix_text, _ = build_training_example(row["prompt"], row["chosen"])
            chosen_vec = mean_completion_activation(
                model=model,
                tokenizer=tokenizer,
                prefix_text=prefix_text,
                completion_text=row["chosen"],
                layer_index=args.layer_index,
                device=device,
            )
            rejected_vec = mean_completion_activation(
                model=model,
                tokenizer=tokenizer,
                prefix_text=prefix_text,
                completion_text=row["rejected"],
                layer_index=args.layer_index,
                device=device,
            )
            diff = chosen_vec - rejected_vec
            diffs.append(diff)
            all_diffs.append(diff)

        stacked = torch.stack(diffs)
        direction = stacked.mean(dim=0)
        norm = float(direction.norm().item())
        axis_vectors[axis_name] = direction

        unit = direction / direction.norm().clamp_min(1e-12)
        mean_projection = float((stacked @ unit).mean().item())
        summary_rows.append(
            {
                "axis": axis_name,
                "num_pairs": len(axis_rows),
                "layer_index": args.layer_index,
                "vector_norm": norm,
                "mean_pair_projection": mean_projection,
            }
        )
        print(
            f"axis={axis_name} pairs={len(axis_rows)} "
            f"vector_norm={norm:.4f} mean_pair_projection={mean_projection:.4f}"
        )

    global_direction = torch.stack(all_diffs).mean(dim=0)
    global_norm = float(global_direction.norm().item())
    summary_payload = {
        "model_name": args.model_name,
        "device": device,
        "layer_index": args.layer_index,
        "axes": summary_rows,
        "global_direction_norm": global_norm,
        "num_total_pairs": len(rows),
    }

    torch.save(
        {
            "model_name": args.model_name,
            "layer_index": args.layer_index,
            "axes": axis_vectors,
            "global": global_direction,
        },
        output_dir / "directions.pt",
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"saved direction tensors to {output_dir / 'directions.pt'}")
    print(f"saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
