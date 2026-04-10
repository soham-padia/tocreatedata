from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanity_direction.data import write_jsonl, write_lines
from humanity_direction.pairs import collect_prompts, load_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged seed-pair and eval-prompt views from axis files.")
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--seed-output", required=True)
    parser.add_argument("--prompts-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_pairs(args.pairs_path)
    prompts = collect_prompts(rows)
    write_jsonl(args.seed_output, rows)
    write_lines(args.prompts_output, prompts)
    print(f"wrote {len(rows)} rows to {args.seed_output}")
    print(f"wrote {len(prompts)} prompts to {args.prompts_output}")


if __name__ == "__main__":
    main()
