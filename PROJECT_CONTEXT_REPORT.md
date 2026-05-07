# Project Context Report

Date: 2026-05-05

This document is meant to be pasted into a fresh chat so the other assistant has enough context to understand the project, the codebase, the current experiments, and the decisions made so far.

## Executive Summary

This repo is a research scaffold for mechanistic dataset construction. The core goal is to find short text sequences that move a language model's internal activations along a direction extracted from pro-human contrastive examples.

At the highest level:

1. Build a pro-human direction from `chosen` vs `rejected` examples across multiple axes.
2. Search for candidate text/token sequences that push the model's hidden states strongly along that direction.
3. Also search with the direction sign flipped, to find sequences that push against the same direction.
4. Treat the resulting top-ranked sequences as a dataset candidate.
5. Study whether those sequences are semantically obvious, repetitive, strange, opaque, or behaviorally meaningful.

The current main model is:

```text
Qwen/Qwen2.5-7B-Instruct
```

The current main direction tensor used for experiments is:

```text
outputs/mechanistic_dataset/pro_human_smoke_20260503_205125/direction/directions.pt
```

The most recent active research path is not the old 15-entry lexicon path. It is a tokenizer-shortlist path:

```text
SEARCH_SPACE=token_file
TOKEN_FILE=data/token_shortlists/qwen25_diverse_5000.jsonl
```

The latest long runs reached depth-2 expansion, but they were stopped as too expensive under the current branching configuration. The next intended run is a faster reduced-branching version, but there is one important implementation caveat: in the current code, `EXPANSION_ALPHABET_SIZE` only limits `tokenizer_vocab` expansion. For `token_file`, expansion currently uses the full shortlist, so reducing `EXPANSION_ALPHABET_SIZE` will not speed up `token_file` runs until the miner is patched.

## Research Hypothesis

The working hypothesis is:

```text
Some short input sequences can strongly move a model along alignment-relevant or misalignment-relevant internal directions, and some of those sequences may be surprising, non-obvious, or semantically opaque to humans.
```

Early evidence:

- The old lexicon search found readable, repetitive pro-social strings such as `truthfulness solidarity cooperation`.
- The tokenizer-vocab and tokenizer-shortlist negative-direction searches found stranger triggers such as `comic`, `.Mock`, `Emoji`, `='')\n`, `!)\n`, `-chan`, and code/newline fragments.
- The negative direction consistently produced larger score magnitudes than the positive direction in the current experiments.

Important interpretation rule:

```text
Negative projection does not automatically mean morally anti-human.
```

It means the sequence moves the measured activation delta against the extracted direction vector. Behavioral validation is still needed.

## Data

Main contrastive data lives in:

```text
data/axes/*.sample.jsonl
```

There are 15 axes:

```text
accountability
boundaries
conflict_resolution
empathy
fairness
feedback
inclusion
integrity
leadership
learning
ownership
privacy
respect
safety
trust
```

Current row counts:

```text
20 rows each for most axes
30 rows for fairness
325 total axis rows
```

Each axis row has:

```json
{
  "axis": "fairness",
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Meaning:

- `chosen` is intended to be more pro-human for that axis.
- `rejected` is intended to be less pro-human for that axis.
- The direction extractor uses the hidden-state difference between these completions.

There are also generated or helper data views:

```text
data/seed_pairs.sample.jsonl
data/eval_prompts.txt
```

Those can be regenerated with:

```bash
python3 scripts/build_dataset_views.py \
  --pairs-path data/axes \
  --seed-output data/seed_pairs.sample.jsonl \
  --prompts-output data/eval_prompts.txt
```

The original hand-written lexicon is:

```text
data/lexicon.txt
```

It has 15 lexicon units:

```text
human dignity
human welfare
human flourishing
respect people
protect agency
cooperation
mutual care
truthfulness
compassion
solidarity
do no dehumanize
do not exploit
life affirming
shared humanity
care with honesty
```

These are lexicon units, not tokenizer tokens. A length-15 lexicon search means 15 lexicon entries, not 15 raw model tokens.

The newer tokenizer shortlist file is:

```text
data/token_shortlists/qwen25_diverse_5000.jsonl
```

It stores exact token IDs, decoded text, bucket labels, and weirdness metadata.

## Prompting Layer

Prompt helpers live in:

```text
src/humanity_direction/prompting.py
```

The built-in system prompt is:

```text
You are a careful assistant. Favor human dignity, cooperation, truthfulness, and preservation of agency.
```

This matters because the baseline and injected prompts already contain a pro-human prior. It is not a neutral prompt environment.

Main helpers:

- `build_chat_prompt(user_prompt, system_prompt=SYSTEM_PROMPT)`
  Builds the chat-style prompt with system, user, and assistant markers.

- `build_injected_prompt(user_prompt, phrase)`
  Older string-based injection helper.

- `build_injected_prompt_parts(user_prompt)`
  Returns a prefix and suffix so the miner can insert exact token IDs between them. This avoids retokenizing a candidate string after it has already been chosen as token IDs.

- `build_training_example(prompt, completion)`
  Builds the prefix and full training text for direction extraction.

The current mechanistic miner uses exact token insertion:

```text
<|system|>
SYSTEM_PROMPT Additional steering phrase: [candidate tokens].
<|user|>
prompt
<|assistant|>
probe_completion
```

The default probe completion is:

```text
I will respond carefully and helpfully.
```

The miner measures the activation over this probe completion, not over a generated completion.

## Direction Extraction

Direction extraction is implemented in:

```text
scripts/extract_mechanistic_directions.py
src/humanity_direction/activations.py
src/humanity_direction/pairs.py
src/humanity_direction/prompting.py
```

Command shape:

```bash
python3 scripts/extract_mechanistic_directions.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --pairs-path data/axes \
  --output-dir outputs/mechanistic_directions/all_axes_layer_minus1 \
  --layer-index -1
```

High-level algorithm:

1. Load all rows from `data/axes`.
2. Optionally filter by `--axis` or `--axes`.
3. Group rows by axis.
4. Load model and tokenizer.
5. For each row:
   - build the prompt prefix
   - run the chosen completion
   - run the rejected completion
   - extract hidden states at `layer_index`
   - average the hidden states over completion tokens
   - compute `diff = h(chosen) - h(rejected)`
6. Average diffs per axis.
7. Average all diffs globally.
8. Save `directions.pt` and `summary.json`.

Low-level activation extraction:

```text
mean_completion_activation(model, tokenizer, prefix_text, completion_text, layer_index, device)
```

This function:

1. Tokenizes the prefix.
2. Tokenizes prefix plus completion.
3. Runs the model with `output_hidden_states=True`.
4. Takes `outputs.hidden_states[layer_index]`.
5. Slices the completion token range.
6. Averages those hidden states.
7. Returns a CPU float tensor.

Saved tensor structure:

```python
{
    "model_name": "...",
    "layer_index": -1,
    "axes": {
        "fairness": tensor(...),
        ...
    },
    "global": tensor(...),
}
```

The current default target is the global direction:

```text
DIRECTION_NAME=global
```

## Mechanistic Scoring

Direction loading and projection scoring live in:

```text
src/humanity_direction/mechanistic_scoring.py
```

`load_direction_vector(path, direction_name)`:

- loads `directions.pt`
- selects either `payload["global"]` or `payload["axes"][direction_name]`
- converts to CPU float
- normalizes the vector
- returns `(unit_vector, metadata)`

The miner applies the sign:

```python
direction_unit = direction_unit * direction_sign
```

So:

- `DIRECTION_SIGN=1` searches with the extracted direction
- `DIRECTION_SIGN=-1` searches against it

Candidate score:

```text
score = mean_over_prompts(dot(h_injected - h_baseline, unit_direction))
```

Where:

- `h_baseline` is the mean suffix activation for the probe completion under the normal prompt.
- `h_injected` is the mean suffix activation for the probe completion when the candidate token sequence is injected into the system prompt.
- The suffix is the fixed probe completion.
- The score is averaged across the train prompt pool.

This is inference only. There is no training:

- no optimizer
- no gradient updates
- no weight updates
- no persistent hidden state across prompts
- each candidate evaluation is a fresh forward pass

## Current Miner

The main miner is:

```text
scripts/mine_pro_human_sequences.py
```

Main config class:

```text
SequenceMiningConfig
```

Important arguments:

```text
--model-name
--direction-tensors
--direction-name
--direction-sign
--output-dir
--search-space lexicon|tokenizer_vocab|token_file
--lexicon-file
--token-file
--prompts-file
--pairs-path
--axis
--axes
--probe-completion
--search-mode exhaustive|beam
--min-phrase-len
--max-phrase-len
--batch-size
--retain-top-k
--beam-width
--expansion-alphabet-size
--validation-fraction
--split-seed
--checkpoint-every-batches
--shard-index
--num-shards
```

Prompt pool behavior:

1. If `--prompts-file` is provided, load prompts from that file.
2. Else if `--pairs-path` is provided, load prompts from the axis pair data.
3. Else fall back to `carrier_prompt`.

Current Explorer launcher defaults to:

```text
PAIRS_PATH=data/axes
VALIDATION_FRACTION=0.2
SPLIT_SEED=0
```

So the prompt pool is split into train and validation prompts. Search uses train prompts. Final retained rows are reranked by validation score when validation prompts exist.

Candidate row fields include:

```text
candidate_key
steering_sentence
unit_sequence
unit_positions
token_ids
token_length
length
mechanistic_score
activation_delta_norm
selection_score
validation_mechanistic_score
validation_activation_delta_norm
direction_name
direction_sign
search_mode
search_space
probe_completion
shard_index
first_unit_index
num_prompts_scored
```

`selection_score` is the ranking field:

- during search, it starts as train `mechanistic_score`
- after validation, it becomes `validation_mechanistic_score`

## Search Spaces

The miner supports three search spaces.

### `lexicon`

Loads entries from `data/lexicon.txt`.

For lexicon mode:

- each search unit is a phrase
- phrases are joined with spaces
- candidate text is retokenized as a normal string

This is useful for controlled phrase-level experiments, but it is biased toward readable pro-social language.

### `tokenizer_vocab`

Loads the model tokenizer vocabulary directly.

For tokenizer-vocab mode:

- special tokens are skipped
- empty or whitespace-only decoded tokens are skipped
- candidates are exact token IDs

This is broad, but the raw Qwen vocab is about 150k tokens. The seed stage was too expensive in practice.

### `token_file`

Loads exact token IDs from a JSONL shortlist, currently:

```text
data/token_shortlists/qwen25_diverse_5000.jsonl
```

This is the current preferred search space because it is much smaller than the full vocabulary while still preserving broad coverage and weird-token candidates.

## Token Shortlist Builder

The shortlist builder is:

```text
scripts/build_token_shortlist.py
sbatch/build_token_shortlist.sbatch
```

Purpose:

- avoid seeding from the full 150k+ Qwen vocabulary
- keep broad embedding-space coverage
- preserve a reserved bucket of weird/code/punctuation tokens for opaque trigger discovery

Default shortlist settings:

```text
SHORTLIST_SIZE=5000
WEIRD_RESERVE=500
CLEAN_POOL_SIZE=20000
WEIRD_POOL_SIZE=5000
PROJECT_DIM=32
SEED=0
```

Token classification:

- drops empty and whitespace-only decoded tokens
- drops special tokens
- marks a token as weird if it has control chars, replacement chars, punctuation-only text, code punctuation, non-ASCII text, or short lowercase fragment-like form
- otherwise marks it as clean

Selection method:

1. Collect clean and weird token candidates.
2. Random-sample a clean pool.
3. Sort/sample a weird pool by weirdness.
4. Load model input embeddings.
5. Normalize embeddings.
6. Random-project embeddings to lower dimension.
7. Use approximate farthest-point sampling for diversity.
8. Select about 4500 clean and 500 weird tokens.
9. Write JSONL and summary JSON.

Output row shape:

```json
{
  "token_id": 123,
  "token": "...",
  "decoded_text": "...",
  "token_ids": [123],
  "bucket": "clean",
  "weird_score": 0,
  "flags": []
}
```

## Search Modes

### Exhaustive

The exhaustive path enumerates sequences up to the requested length.

It shards by first unit:

```text
unit.index % NUM_SHARDS == SHARD_INDEX
```

This is feasible only for small search spaces and small lengths.

### Beam

The beam path is the main path for larger searches.

Current beam logic:

1. Score shard-local single-unit seeds.
2. Keep the top `BEAM_WIDTH` seed candidates as the frontier.
3. Expand each frontier candidate with an expansion alphabet.
4. Score proposals.
5. Keep top `BEAM_WIDTH` as the next frontier.
6. Keep top `RETAIN_TOP_K` overall candidates.
7. Repeat until `MAX_PHRASE_LEN`.

Important implementation caveat:

```text
EXPANSION_ALPHABET_SIZE currently only limits expansion for search_space=tokenizer_vocab.
```

Current code:

```python
if config.search_space == "tokenizer_vocab":
    expansion_positions = dedupe_preserve_order(
        [candidate.unit_positions[0] for candidate in frontier_candidates[: config.expansion_alphabet_size]]
    )
else:
    expansion_positions = list(range(len(units)))
```

Therefore, for `search_space=token_file`, the miner currently expands against the entire token shortlist, not just `EXPANSION_ALPHABET_SIZE`.

This explains why the recent `token_file` length-2 runs remained expensive even though `EXPANSION_ALPHABET_SIZE=64` was set.

Before running the planned faster token-shortlist experiment, the miner should be patched so `token_file` also respects `EXPANSION_ALPHABET_SIZE`, or a separate option should be added to choose the expansion subset.

## Checkpointing and Timeout Behavior

The miner is timeout-safe for Slurm in the practical sense.

Checkpoint files:

```text
checkpoint.json
retained_candidates.partial.jsonl
summary.partial.json
```

These are written under:

```text
outputs/mechanistic_dataset/<run_root>/shards/shard_N/
```

The shard sbatch script requests:

```text
#SBATCH --signal=TERM@120
```

So Slurm sends `TERM` 120 seconds before hard kill. The Python miner catches this and writes a checkpoint before exiting.

Checkpoint compatibility checks include:

```text
search_space
lexicon_file
token_file
search_mode
min_phrase_len
max_phrase_len
beam_width
expansion_alphabet_size
retain_top_k
validation_fraction
split_seed
shard_index
num_shards
pairs_path
prompts_file
axis
axes
direction_name
direction_sign
probe_completion
```

To resume a run:

- use the same `RUN_ROOT`
- keep the same search config
- change only wall time or batch-related runtime settings

If config does not match, resume fails intentionally.

OOM behavior:

- `compute_suffix_activations_batched` catches CUDA OOM when possible
- halves the effective micro-batch
- clears CUDA cache
- retries
- this protects long runs from crashing due to one oversized batch

## Merge

Merge script:

```text
scripts/merge_mechanistic_sequences.py
sbatch/merge_mechanistic_sequences.sbatch
```

Merge behavior:

1. Find `shards/shard_*/retained_candidates.jsonl`.
2. Load all rows.
3. Deduplicate by `candidate_key` when available, else `steering_sentence`.
4. For duplicates, keep the row with higher `selection_score`.
5. Sort by `selection_score`, then train `mechanistic_score`.
6. Write:
   - `top_sentences.jsonl`
   - `summary.json`

The merge computation is CPU-style JSON sorting, but Explorer's `gpu` partition rejected jobs without a GPU request. For cluster policy reasons, the merge sbatch currently requests `gpu:v100-sxm2:1` even though the script itself does not use CUDA.

## Reporting

There are two reporting paths.

### Full result summarizer

```text
scripts/summarize_mechanistic_results.py
```

This is for completed runs with merged outputs. It can summarize one run or compare two runs.

It reports:

- score distributions
- activation delta norm distributions
- length distributions
- top-slice means
- repetition/collapse metrics
- unit frequencies
- bigrams/trigrams
- top examples
- overlap and distinctive units across runs

### Progress reporter

```text
scripts/report_mechanistic_progress.py
```

This is for completed or partial checkpointed runs. It produces paste-friendly markdown from `summary.json`, `summary.partial.json`, `retained_candidates.jsonl`, or `retained_candidates.partial.jsonl`.

Use it when runs timed out but checkpointed successfully.

Example:

```bash
python3 scripts/report_mechanistic_progress.py \
  --run-root outputs/mechanistic_dataset/qwen_shortlist_pos_20260504_191333 \
  --run-root outputs/mechanistic_dataset/qwen_shortlist_neg_20260504_191347 \
  --output-file temp_cache.md
```

## HPC Workflow

Main launcher:

```text
run_pro_human_mechanistic_hpc.sh
```

Relevant sbatch files:

```text
sbatch/extract_mechanistic_directions.sbatch
sbatch/mine_pro_human_sequences_shard.sbatch
sbatch/merge_mechanistic_sequences.sbatch
sbatch/build_token_shortlist.sbatch
```

Current GPU request:

```text
#SBATCH --gres=gpu:v100-sxm2:1
```

Reason:

- user prefers not to jump to the highest-end GPU for environmental reasons
- `v100-sxm2` is a middle ground
- previous smaller GPU had about 16 GB and OOMed
- `v100-sxm2` gives 32 GB and was enough to run the current search with adaptive batching

Launcher behavior:

- creates or reuses `RUN_ROOT`
- extracts directions if `DIRECTION_TENSORS` is not provided
- submits `NUM_SHARDS` shard jobs
- submits merge with `afterok` dependency on all shards
- cancels already-submitted jobs if submission fails partway through
- supports resume by passing an existing `RUN_ROOT`

## Completed Earlier Experiments

### Smoke extraction and lexicon search

Run root:

```text
outputs/mechanistic_dataset/pro_human_smoke_20260503_205125
```

This produced a usable direction file:

```text
outputs/mechanistic_dataset/pro_human_smoke_20260503_205125/direction/directions.pt
```

This direction is reused by later experiments.

### Lexicon beam-15 positive run

Run root:

```text
outputs/mechanistic_dataset/pro_human_beam15_20260503_233034
```

Configuration:

```text
SEARCH_SPACE=lexicon
SEARCH_MODE=beam
DIRECTION_SIGN=1
MAX_PHRASE_LEN=15
BEAM_WIDTH=64
```

Result:

- completed
- produced `top_sentences.jsonl` with 1000 rows
- top results were readable but repetitive

Examples:

```text
truthfulness solidarity cooperation cooperation cooperation human dignity ...
cooperation solidarity truthfulness
cooperation truthfulness cooperation
```

Interpretation:

- pipeline worked end to end
- top results were mostly obvious pro-social motifs
- search collapsed into repeated terms

### Lexicon beam-15 negative run

Run root:

```text
outputs/mechanistic_dataset/anti_human_beam15_20260503_233047
```

Configuration:

```text
SEARCH_SPACE=lexicon
SEARCH_MODE=beam
DIRECTION_SIGN=-1
MAX_PHRASE_LEN=15
BEAM_WIDTH=64
```

Result:

- completed
- negative direction did not produce semantically evil text
- it produced phrases such as `protect agency`, `do not exploit`, and `respect people`

Interpretation:

- sign flip works geometrically
- negative projection does not equal ordinary semantic anti-human meaning
- the small pro-social lexicon constrained the output space too strongly

## Current Token-Shortlist Runs

Current positive run root:

```text
outputs/mechanistic_dataset/qwen_shortlist_pos_20260504_191333
```

Current negative run root:

```text
outputs/mechanistic_dataset/qwen_shortlist_neg_20260504_191347
```

Shared config:

```text
SEARCH_SPACE=token_file
TOKEN_FILE=data/token_shortlists/qwen25_diverse_5000.jsonl
SEARCH_MODE=beam
NUM_SHARDS=2
MIN_PHRASE_LEN=1
MAX_PHRASE_LEN=2
BEAM_WIDTH=32
EXPANSION_ALPHABET_SIZE=64
BATCH_SIZE=32
RETAIN_TOP_K=2000
FINAL_TOP_K=500
VALIDATION_FRACTION=0.2
```

1-hour runs timed out during seed, but checkpointed successfully.

7-hour resumed runs:

- loaded checkpoints
- completed seed
- entered depth-2 expansion
- timed out again during expansion

Latest compact status:

Positive shard 0:

```text
phase=expand
depth=2
scored=7980
retained=2000
best=12.701266288757324
```

Positive shard 1:

```text
phase=expand
depth=2
scored=7964
retained=2000
best=12.228668212890625
```

Negative shard 0:

```text
phase=expand
depth=2
scored=7820
retained=2000
best=78.66780090332031
```

Negative shard 1:

```text
phase=expand
depth=2
scored=7804
retained=2000
best=69.893310546875
```

Representative positive frontier:

```text
responding
commit
permission
.Accept
.respond
request
Response
communicated
```

Representative negative frontier:

```text
Sass
.Mock
!)\n
uh
rapper
='')\n
comic
Emoji
-chan
){\n
```

Interpretation:

- tokenizer-shortlist search is doing what it was meant to do
- positive side looks procedural, response-like, and commitment-oriented
- negative side looks stranger, stylistic, persona-like, and artifact-like
- negative score magnitudes remain much larger than positive magnitudes
- this supports the idea that non-obvious internal triggers exist, especially in the negative direction

## Why We Stopped Continuing That Exact Run

The current run is not conceptually blocked. It reached the important depth-2 stage.

The problem is cost:

- the 1-hour runs timed out in seed
- the 7-hour resumes timed out in expand
- the run still had not merged final top-500 outputs

The deeper reason is the implementation caveat:

```text
For token_file, expansion uses all 5000 shortlist units.
```

So the depth-2 proposal space is roughly:

```text
BEAM_WIDTH * 5000
```

With `BEAM_WIDTH=32`, that is about:

```text
160,000 proposals per shard
```

This is much larger than the intended:

```text
BEAM_WIDTH * EXPANSION_ALPHABET_SIZE
32 * 64 = 2048 proposals per shard
```

That explains why the search did not finish.

## Immediate Recommended Fix

Before launching the next reduced-branching token-shortlist run, patch `scripts/mine_pro_human_sequences.py` so `token_file` respects `EXPANSION_ALPHABET_SIZE`.

Current behavior:

```python
if config.search_space == "tokenizer_vocab":
    expansion_positions = dedupe_preserve_order(
        [candidate.unit_positions[0] for candidate in frontier_candidates[: config.expansion_alphabet_size]]
    )
else:
    expansion_positions = list(range(len(units)))
```

Recommended behavior:

```python
if config.search_space in {"tokenizer_vocab", "token_file"}:
    expansion_positions = dedupe_preserve_order(
        [candidate.unit_positions[0] for candidate in frontier_candidates[: config.expansion_alphabet_size]]
    )
else:
    expansion_positions = list(range(len(units)))
```

However, this exact simple patch only expands using positions already present in the beam frontier. For a richer token-file expansion, a better design is:

```text
1. score seed tokens
2. keep top BEAM_WIDTH as frontier
3. keep top EXPANSION_ALPHABET_SIZE seed tokens as expansion alphabet
4. expand frontier by that alphabet
```

That requires retaining a separate top seed alphabet, not just using the `BEAM_WIDTH` frontier rows.

Practical recommendation:

- first make `token_file` expansion bounded
- use the top seed rows as the expansion alphabet
- then run a cleaner length-2 experiment

## Recommended Next Experiment

After patching token-file expansion behavior, run:

```text
MAX_PHRASE_LEN=2
BEAM_WIDTH=32
EXPANSION_ALPHABET_SIZE=16
NUM_SHARDS=2
BATCH_SIZE=32
VALIDATION_FRACTION=0.2
```

Positive command shape:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
DIRECTION_TENSORS=outputs/mechanistic_dataset/pro_human_smoke_20260503_205125/direction/directions.pt \
RUN_SLUG=qwen_shortlist_pos_fast \
SEARCH_SPACE=token_file \
TOKEN_FILE=data/token_shortlists/qwen25_diverse_5000.jsonl \
SEARCH_MODE=beam \
NUM_SHARDS=2 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=2 \
BEAM_WIDTH=32 \
EXPANSION_ALPHABET_SIZE=16 \
BATCH_SIZE=32 \
RETAIN_TOP_K=2000 \
FINAL_TOP_K=500 \
VALIDATION_FRACTION=0.2 \
SHARD_TIME=03:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

Negative command shape:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
DIRECTION_TENSORS=outputs/mechanistic_dataset/pro_human_smoke_20260503_205125/direction/directions.pt \
RUN_SLUG=qwen_shortlist_neg_fast \
SEARCH_SPACE=token_file \
TOKEN_FILE=data/token_shortlists/qwen25_diverse_5000.jsonl \
SEARCH_MODE=beam \
DIRECTION_SIGN=-1 \
NUM_SHARDS=2 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=2 \
BEAM_WIDTH=32 \
EXPANSION_ALPHABET_SIZE=16 \
BATCH_SIZE=32 \
RETAIN_TOP_K=2000 \
FINAL_TOP_K=500 \
VALIDATION_FRACTION=0.2 \
SHARD_TIME=03:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

If still too slow:

```text
BEAM_WIDTH=16
EXPANSION_ALPHABET_SIZE=16
```

## Older Baseline Path

The repo still contains an older behavioral/rubric-based mining path.

Files:

```text
src/humanity_direction/direction.py
src/humanity_direction/search.py
scripts/score_seed_pairs.py
scripts/mine_candidates.py
scripts/mine_mechanistic_dataset.py
```

This path:

1. Loads a lexical direction spec from `data/direction_spec.sample.json`.
2. Scores completions by positive and negative cue hits.
3. Generates completions with or without injected phrases.
4. Scores whether the phrase improved the generated completion according to the rubric.
5. Uses a simple phrase beam search.

This is now baseline-only. It is useful for comparison, but it is not the primary research objective.

## Key Files by Responsibility

### Core data utilities

```text
src/humanity_direction/data.py
```

- `load_jsonl`
- `load_lines`
- `write_jsonl`
- `write_lines`

```text
src/humanity_direction/pairs.py
```

- `load_pairs`
- `collect_prompts`

### Prompting

```text
src/humanity_direction/prompting.py
```

- defines `SYSTEM_PROMPT`
- builds baseline prompts
- builds injected prompt parts
- builds training examples

### Activation extraction

```text
src/humanity_direction/activations.py
```

- loads model/tokenizer
- encodes text
- extracts terminal activations
- extracts mean completion activations
- extracts batched mean suffix activations from exact token IDs

### Direction extraction

```text
scripts/extract_mechanistic_directions.py
```

- builds per-axis vectors
- builds global vector
- writes `directions.pt`
- writes extraction summary

### Direction loading/scoring

```text
src/humanity_direction/mechanistic_scoring.py
```

- loads direction tensors
- normalizes vectors
- computes projection score

### Search/mining

```text
scripts/mine_pro_human_sequences.py
```

- loads prompt pool
- splits train/validation prompts
- loads search units
- builds exact token candidate sequences
- computes baseline and injected suffix activations
- scores candidate deltas by projection
- supports exhaustive and beam search
- checkpoints partial progress
- validates retained rows
- writes shard outputs

### Token shortlist

```text
scripts/build_token_shortlist.py
```

- filters tokenizer vocab
- separates clean and weird candidates
- uses embedding-space diversity
- writes exact-token shortlist JSONL

### Merge

```text
scripts/merge_mechanistic_sequences.py
```

- combines shard retained rows
- deduplicates
- sorts by `selection_score`
- writes final top dataset

### Reporting

```text
scripts/summarize_mechanistic_results.py
scripts/report_mechanistic_progress.py
```

- summarize completed results
- summarize partial checkpointed runs

### HPC

```text
run_pro_human_mechanistic_hpc.sh
sbatch/*.sbatch
```

- submit extraction, shard mining, merge, and shortlist jobs
- use Explorer modules and conda env
- request `v100-sxm2`
- support checkpoint/resume

## Known Caveats

1. `token_file` expansion currently ignores `EXPANSION_ALPHABET_SIZE`.

This is the most important engineering issue before the next fast run.

2. The built-in system prompt is already pro-human.

This means activation effects are measured in a context with a pro-human prior, not a neutral context.

3. The probe completion may bias scores.

The default probe completion is:

```text
I will respond carefully and helpfully.
```

Tokens related to `will`, `respond`, or helper-style behavior may be advantaged.

4. Negative projection is not behaviorally validated.

The negative search finds sequences that move against the extracted vector. It does not prove those sequences cause harmful behavior.

5. Current datasets are ranked sequence sets, not full downstream fine-tuning datasets.

The output is `top_sentences.jsonl`. A later step is needed to convert these into a richer training or evaluation dataset.

6. Merge currently asks for a GPU due to partition policy.

The merge script itself is CPU-style JSON processing, but Explorer rejected GPU-partition jobs without `--gres`, so the sbatch uses `gpu:v100-sxm2:1`.

## What A New Chat Should Do Next

Recommended next steps:

1. Patch token-file beam expansion so `EXPANSION_ALPHABET_SIZE` actually bounds expansion for `SEARCH_SPACE=token_file`.
2. Run a small local/static verification that checkpoint compatibility still works.
3. Submit positive and negative fast length-2 shortlist runs with `EXPANSION_ALPHABET_SIZE=16`.
4. Generate a progress report with `scripts/report_mechanistic_progress.py`.
5. If merged results complete, run `scripts/summarize_mechanistic_results.py` to compare positive vs negative.
6. Evaluate whether the top negative triggers are real behaviorally or just projection artifacts.

The important conceptual state is:

```text
The project is working mechanistically, but the search strategy needs bounded branching before longer or cleaner tokenizer-level runs are practical.
```
