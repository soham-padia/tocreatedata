# Humanity Direction Mechanistic Mining

This project mines short text sequences that push a language model's **internal activations** along a target direction.

The current primary target is a merged **pro-human** direction extracted from contrastive `chosen` vs `rejected` pairs across these axes:

- fairness
- respect
- accountability
- empathy
- boundaries
- integrity
- inclusion
- conflict resolution
- ownership
- trust
- feedback
- leadership
- learning
- privacy
- safety

The older rubric-based prompt-steering pipeline is still in the repo, but it is now **baseline-only**. The current research objective is mechanistic.

## Core Idea

For each training pair:

- `chosen` = more pro-human completion
- `rejected` = less pro-human completion

We extract hidden activations for both and compute a difference vector:

`d_i = h(chosen_i) - h(rejected_i)`

Then we average those vectors:

- per axis
- and globally across all axes

That gives an internal direction vector `d`.

For any candidate sequence:

1. build a baseline prompt activation `h_base`
2. inject the candidate sequence and get `h_candidate`
3. compute `delta = h_candidate - h_base`
4. score the candidate by projection onto the direction:

`score = dot(delta, unit(d))`

Interpretation:

- positive score = moves with the extracted direction
- negative score = moves against it
- larger magnitude = stronger mechanistic effect

Important: a negative score means "opposite to the extracted direction vector," not automatically "anti-human" in the ordinary semantic sense.

## Current Hypothesis

The working hypothesis is:

There exists a subset of short text sequences that strongly move the model along alignment-relevant or misalignment-relevant internal directions, and some of those sequences may be semantically opaque, repetitive, or surprising to humans even though they are mechanistically effective.

## Repository Layout

### Data

- `data/axes/*.sample.jsonl`
  Contrastive prompt/completion pairs grouped by axis.
- `data/seed_pairs.sample.jsonl`
  Merged seed-pair view generated from `data/axes/`.
- `data/eval_prompts.txt`
  Prompt-only view generated from `data/axes/`.
- `data/lexicon.txt`
  Search alphabet for sequence mining. These are **lexicon units**, not raw tokenizer tokens.
- `data/direction_spec.sample.json`
  Weighted lexical rubric used by the older baseline pipeline.

### Mechanistic path

- `scripts/extract_mechanistic_directions.py`
  Extracts per-axis and global direction vectors from `chosen` vs `rejected` pairs.
- `scripts/mine_pro_human_sequences.py`
  Mines sequences by mechanistic projection score.
- `scripts/merge_mechanistic_sequences.py`
  Merges retained shard outputs into one final top-K dataset.
- `scripts/summarize_mechanistic_results.py`
  Produces detailed JSON and Markdown summaries for one or more mechanistic runs.
- `src/humanity_direction/activations.py`
  Hidden-state extraction utilities.
- `src/humanity_direction/mechanistic_scoring.py`
  Direction loading and projection scoring.
- `run_pro_human_mechanistic_hpc.sh`
  "Submit and forget" Explorer launcher for mechanistic mining.

### Older baseline path

- `scripts/score_seed_pairs.py`
  Checks whether the lexical rubric prefers `chosen` over `rejected`.
- `scripts/mine_candidates.py`
  Mines phrases by lexical rubric improvement.
- `run_overnight_hpc_fairness_conda.sh`
- `run_all_axes_hpc_conda.sh`

These are useful as behavioral baselines, but they are not the main objective anymore.

## Data Format

Each row in `data/axes/*.sample.jsonl` contains:

- `prompt`
- `chosen`
- `rejected`
- `axis`
- optional `notes`

The `chosen` response should be more aligned with the intended axis than the `rejected` response.

If you edit the axis files, rebuild the merged views:

```bash
python3 scripts/build_dataset_views.py \
  --pairs-path data/axes \
  --seed-output data/seed_pairs.sample.jsonl \
  --prompts-output data/eval_prompts.txt
```

## Mechanistic Workflow

### 1. Extract direction vectors

Local or interactive run:

```bash
python3 scripts/extract_mechanistic_directions.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --pairs-path data/axes \
  --output-dir outputs/mechanistic_directions/all_axes_layer_minus1 \
  --layer-index -1
```

Outputs:

- `directions.pt`
- `summary.json`

`directions.pt` contains:

- one vector per axis
- one merged global vector

### 2. Mine sequences by projection score

#### Exhaustive search

Use this for short sequence lengths over the controlled lexicon:

```bash
python3 scripts/mine_pro_human_sequences.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --direction-tensors outputs/mechanistic_directions/all_axes_layer_minus1/directions.pt \
  --direction-name global \
  --direction-sign 1 \
  --lexicon-file data/lexicon.txt \
  --output-dir outputs/mechanistic_dataset/local_exhaustive \
  --search-mode exhaustive \
  --min-phrase-len 1 \
  --max-phrase-len 3 \
  --batch-size 8 \
  --retain-top-k 4000
```

#### Heuristic beam search

Use this for broader phrase-level search, including longer sequences:

```bash
python3 scripts/mine_pro_human_sequences.py \
  --model-name "Qwen/Qwen2.5-7B-Instruct" \
  --direction-tensors outputs/mechanistic_directions/all_axes_layer_minus1/directions.pt \
  --direction-name global \
  --direction-sign 1 \
  --lexicon-file data/lexicon.txt \
  --output-dir outputs/mechanistic_dataset/local_beam15 \
  --search-mode beam \
  --min-phrase-len 1 \
  --max-phrase-len 15 \
  --beam-width 64 \
  --batch-size 32 \
  --retain-top-k 5000
```

Important:

- `max-phrase-len=15` means **15 lexicon units**
- it does **not** mean 15 raw tokenizer tokens

### 3. Flip the direction sign

To mine the opposite-projection dataset, keep everything else fixed and flip:

```bash
--direction-sign -1
```

That produces sequences that move against the extracted direction vector.

This is a geometric opposite, not automatically a behavioral or moral opposite.

### 4. Summarize results

Single run:

```bash
python3 scripts/summarize_mechanistic_results.py \
  --run-root outputs/mechanistic_dataset/local_beam15
```

Compare two runs:

```bash
python3 scripts/summarize_mechanistic_results.py \
  --run-root outputs/mechanistic_dataset/pro_human_beam15_... \
  --label pro_human \
  --run-root outputs/mechanistic_dataset/anti_human_beam15_... \
  --label negative_projection
```

Outputs:

- `report.json`
- `report.md`

The summarizer reports:

- score distribution
- activation-delta distribution
- length distribution
- repetition/collapse metrics
- top units, bigrams, and trigrams
- exact overlap across two runs
- distinctive-unit comparison across two runs

## Explorer HPC Workflow

The recommended Explorer path is:

1. build the conda environment on a `gpu-interactive` node
2. run `bash update.sh` on the login node
3. submit mechanistic jobs from the login node

For the main mechanistic launcher:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
RUN_UPDATE=1 \
RUN_SLUG=pro_human_global \
NUM_SHARDS=1 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=3 \
BATCH_SIZE=8 \
RETAIN_TOP_K=4000 \
FINAL_TOP_K=1000 \
SHARD_TIME=01:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

For a broader heuristic search:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
DIRECTION_TENSORS=outputs/mechanistic_dataset/pro_human_smoke_.../direction/directions.pt \
RUN_SLUG=pro_human_beam15 \
DIRECTION_SIGN=1 \
SEARCH_MODE=beam \
NUM_SHARDS=1 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=15 \
BEAM_WIDTH=64 \
BATCH_SIZE=32 \
RETAIN_TOP_K=5000 \
FINAL_TOP_K=1000 \
SHARD_TIME=01:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

The launcher will:

1. optionally extract the direction if `DIRECTION_TENSORS` is not provided
2. submit one or more shard jobs
3. submit a merge job
4. write outputs under `outputs/mechanistic_dataset/`

See [sbatch/README.md](/Users/sohampadia/workspace/Nikhil/research/tocreatedata/sbatch/README.md) for the job files.

## Current Interpretation Rules

When inspecting results:

- top `+1` sequences are the strongest movers **with** the extracted direction
- top `-1` sequences are the strongest movers **against** the extracted direction

Do not assume the `-1` dataset is "evil" or "anti-human" just because it is opposite in projection space.

If a semantically positive phrase appears in the negative-projection dataset, that means:

- the phrase moves the chosen layer/state opposite to the extracted global vector
- not that the phrase is morally bad in ordinary language

This is one of the main reasons the mechanistic path is interesting: internal geometry can diverge from naive surface semantics.

## Current Limits

- The search alphabet is still a small hand-written lexicon.
- Long beam searches can collapse into repetitive motifs.
- One extracted direction vector can mix content, tone, style, and generic assistant behavior.
- Negative projection is not yet validated as negative downstream behavior.
- Sequence length is measured in lexicon units, not raw model tokens.

## Near-Term Experiments

- validate top `+1` vs top `-1` sequences on downstream behavioral prompts
- diversify the retained top-K so it is not dominated by near-duplicates
- expand beyond the current moral lexicon to test for genuinely opaque sequences
- compare multiple layers or activation readouts for direction extraction
- test whether the extracted direction is stable across carrier prompts
