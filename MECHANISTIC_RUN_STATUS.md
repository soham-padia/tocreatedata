# Mechanistic Run Status

Date: 2026-05-05

## Purpose

This note records the current stopping point for the tokenizer-shortlist mechanistic search and the decision about what to do next.

It exists so we do not lose the operational context from the recent Explorer runs.

## Current experiment

Model:
- `Qwen/Qwen2.5-7B-Instruct`

Direction source:
- `outputs/mechanistic_dataset/pro_human_smoke_20260503_205125/direction/directions.pt`

Search space:
- `token_file`
- `data/token_shortlists/qwen25_diverse_5000.jsonl`

Search mode:
- `beam`

Shared search settings:
- `NUM_SHARDS=2`
- `MIN_PHRASE_LEN=1`
- `MAX_PHRASE_LEN=2`
- `BEAM_WIDTH=32`
- `EXPANSION_ALPHABET_SIZE=64`
- `BATCH_SIZE=32`
- `RETAIN_TOP_K=2000`
- `FINAL_TOP_K=500`
- `VALIDATION_FRACTION=0.2`

Run roots:
- positive: `outputs/mechanistic_dataset/qwen_shortlist_pos_20260504_191333`
- negative: `outputs/mechanistic_dataset/qwen_shortlist_neg_20260504_191347`

## What happened

### 1. Full tokenizer-vocab search was too expensive

We first tried direct tokenizer-vocab mining and found that the seed stage alone was too slow to be practical. That led to the shortlist approach.

### 2. We switched to a 5000-token shortlist

The shortlist intentionally mixes:
- broad embedding-space coverage
- a reserved bucket of weird/code/punctuation-like tokens

This keeps the search broad enough to test the "interesting trigger" hypothesis without paying the full 150k+ token cost.

### 3. The 1-hour shortlist runs were not enough

The initial 1-hour positive and negative shortlist runs both timed out during `seed`.

Observed seed throughput:
- about `992` candidates per hour per shard

That was enough to show the shortlist was working, but not enough to finish the seed phase.

### 4. We resumed with 7-hour shard windows

The resumed runs successfully:
- loaded checkpoints
- finished `seed`
- entered `expand`
- reached real 2-token search

This was the first point where the search was no longer blocked on single-token ranking.

## Current stopping point

Latest observed partial summaries:

Positive:
- `qwen_shortlist_pos_20260504_191333/shard_0`
  - `phase=expand`
  - `depth=2`
  - `scored=7980`
  - `retained=2000`
  - `best=12.701266288757324`
- `qwen_shortlist_pos_20260504_191333/shard_1`
  - `phase=expand`
  - `depth=2`
  - `scored=7964`
  - `retained=2000`
  - `best=12.228668212890625`

Negative:
- `qwen_shortlist_neg_20260504_191347/shard_0`
  - `phase=expand`
  - `depth=2`
  - `scored=7820`
  - `retained=2000`
  - `best=78.66780090332031`
- `qwen_shortlist_neg_20260504_191347/shard_1`
  - `phase=expand`
  - `depth=2`
  - `scored=7804`
  - `retained=2000`
  - `best=69.893310546875`

Representative frontier behavior:

Positive frontier looks mostly procedural / response-oriented:
- `responding`
- `commit`
- `permission`
- `.Accept`
- `.respond`
- `request`
- `Response`
- `communicated`

Negative frontier looks much stranger and more artifact-like:
- `Sass`
- `.Mock`
- `!)\n`
- `uh`
- `rapper`
- `='')\n`
- `comic`
- `Emoji`
- `-chan`
- `){\n`

## Interpretation

What is now established:
- checkpoint/resume works
- shortlist search works
- the search reached real depth-2 expansion
- positive and negative directions continue to separate strongly
- the negative direction is still producing larger-magnitude and more unusual triggers than the positive direction

What is not established yet:
- final merged top-500 datasets for these exact runs
- whether this exact branching configuration is worth finishing

## Decision

We are stopping brute-force continuation of this exact configuration here.

Specifically, we are not treating the following configuration as the default forward path anymore:
- `MAX_PHRASE_LEN=2`
- `BEAM_WIDTH=32`
- `EXPANSION_ALPHABET_SIZE=64`
- `NUM_SHARDS=2`
- `BATCH_SIZE=32`
- `v100-sxm2`

Reason:
- it is too expensive for the amount of incremental information it provides
- even after checkpointed resumes, the runs were still timing out in `expand`
- the frontier is already informative enough to justify changing the search shape

## Next run policy

The next run should be a faster version of the same shortlist-based length-2 search, not another brute-force continuation of the current branching settings.

Important implementation caveat discovered after this note was first written:
- in the current miner, `EXPANSION_ALPHABET_SIZE` only bounds expansion for `SEARCH_SPACE=tokenizer_vocab`
- for `SEARCH_SPACE=token_file`, expansion currently uses the full shortlist
- therefore lowering `EXPANSION_ALPHABET_SIZE` will not speed up token-file runs until `scripts/mine_pro_human_sequences.py` is patched

Keep:
- same model
- same direction tensors
- same shortlist file
- same sign-split setup
- same `MAX_PHRASE_LEN=2`

After patching token-file expansion, reduce:
- `EXPANSION_ALPHABET_SIZE`
- and, if needed, `BEAM_WIDTH`

Default recommendation for the next run:
- `BEAM_WIDTH=32`
- `EXPANSION_ALPHABET_SIZE=16`

If that is still too slow:
- `BEAM_WIDTH=16`
- `EXPANSION_ALPHABET_SIZE=16`

## Why this is the right tradeoff

The current runs already demonstrated the important qualitative result:
- interesting, non-obvious negative-direction triggers appear in the tokenizer-shortlist search once the run reaches `expand`

At this point, the main bottleneck is search cost, not conceptual uncertainty.

So the correct next move is to shrink branching and finish cleaner runs, rather than spend more cluster time brute-forcing the current configuration.
