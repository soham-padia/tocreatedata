#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_FILE RUN_ROOT [RUN_ROOT ...]" >&2
  exit 1
fi

OUTPUT_FILE="$1"
shift

mkdir -p "$(dirname "$OUTPUT_FILE")"

{
  echo "# Mechanistic Run Export"
  echo
  echo "Generated: $(date)"
  echo
} > "$OUTPUT_FILE"

for RUN_ROOT in "$@"; do
  if [[ ! -d "$RUN_ROOT" ]]; then
    {
      echo "## Run: $RUN_ROOT"
      echo
      echo "Missing run directory."
      echo
    } >> "$OUTPUT_FILE"
    continue
  fi

  SUMMARY_FILE=""
  CANDIDATE_FILE=""
  CHECKPOINT_FILE=""

  if [[ -f "$RUN_ROOT/summary.json" ]]; then
    SUMMARY_FILE="$RUN_ROOT/summary.json"
  elif [[ -f "$RUN_ROOT/shards/shard_0/summary.partial.json" ]]; then
    SUMMARY_FILE="$RUN_ROOT/shards/shard_0/summary.partial.json"
  fi

  if [[ -f "$RUN_ROOT/top_sentences.jsonl" ]]; then
    CANDIDATE_FILE="$RUN_ROOT/top_sentences.jsonl"
  elif [[ -f "$RUN_ROOT/shards/shard_0/retained_candidates.partial.jsonl" ]]; then
    CANDIDATE_FILE="$RUN_ROOT/shards/shard_0/retained_candidates.partial.jsonl"
  fi

  if [[ -f "$RUN_ROOT/shards/shard_0/checkpoint.json" ]]; then
    CHECKPOINT_FILE="$RUN_ROOT/shards/shard_0/checkpoint.json"
  fi

  {
    echo "## Run: $RUN_ROOT"
    echo
    echo '```text'
    find "$RUN_ROOT" -maxdepth 3 -type f | sort
    echo '```'
    echo
  } >> "$OUTPUT_FILE"

  if [[ -n "$SUMMARY_FILE" ]]; then
    {
      echo "### Summary"
      echo
      echo "Path: \`$SUMMARY_FILE\`"
      echo
      echo '```json'
      cat "$SUMMARY_FILE"
      echo
      echo '```'
      echo
    } >> "$OUTPUT_FILE"
  fi

  if [[ -n "$CHECKPOINT_FILE" ]]; then
    {
      echo "### Checkpoint"
      echo
      echo "Path: \`$CHECKPOINT_FILE\`"
      echo
      echo '```json'
      cat "$CHECKPOINT_FILE"
      echo
      echo '```'
      echo
    } >> "$OUTPUT_FILE"
  fi

  if [[ -n "$CANDIDATE_FILE" ]]; then
    {
      echo "### Candidate Rows"
      echo
      echo "Path: \`$CANDIDATE_FILE\`"
      echo
      echo "Line count: $(wc -l < "$CANDIDATE_FILE")"
      echo
      echo '```jsonl'
      head -n 20 "$CANDIDATE_FILE"
      echo
      echo '```'
      echo
    } >> "$OUTPUT_FILE"
  fi
done

echo "wrote $OUTPUT_FILE"
