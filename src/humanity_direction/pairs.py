from __future__ import annotations

from pathlib import Path

from .data import load_jsonl


def load_pairs(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if input_path.is_file():
        return load_jsonl(input_path)

    if not input_path.is_dir():
        raise FileNotFoundError(f"pairs path not found: {input_path}")

    rows: list[dict] = []
    for child in sorted(input_path.glob("*.jsonl")):
        rows.extend(load_jsonl(child))
    return rows


def collect_prompts(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    prompts: list[str] = []
    for row in rows:
        prompt = row["prompt"].strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts
