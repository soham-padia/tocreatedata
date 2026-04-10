from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.data import load_jsonl
from humanity_direction.direction import load_direction_spec, score_text_against_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score chosen/rejected seed pairs against the direction rubric.")
    parser.add_argument("--direction-file", required=True)
    parser.add_argument("--pairs-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direction = load_direction_spec(args.direction_file)
    rows = load_jsonl(args.pairs_file)

    wins = 0
    for index, row in enumerate(rows, start=1):
        chosen = score_text_against_direction(row["chosen"], direction)
        rejected = score_text_against_direction(row["rejected"], direction)
        delta = chosen.total - rejected.total
        if delta > 0:
            wins += 1
        print(
            f"{index}. axis={row.get('axis', 'n/a')} "
            f"chosen={chosen.total:.2f} rejected={rejected.total:.2f} delta={delta:.2f}"
        )

    print(f"direction rubric preferred {wins}/{len(rows)} chosen completions")


if __name__ == "__main__":
    main()
