# File Guide And Project Architecture

This document explains the repository at the file level.

It has four goals:

1. describe what each project file is for
2. summarize the important functions or entrypoints inside each Python file
3. show how data and control flow through the project
4. distinguish the **current mechanistic path** from the older **rubric baseline path**

This guide covers the project files in the repo root, `data/`, `scripts/`, `src/`, and `sbatch/`.
It intentionally ignores:

- `.git/`
- `.venv/`
- `__pycache__/`
- `sbatch/logs/`

## 1. Project In One View

The repository has two research paths:

1. **Current main path: mechanistic direction mining**
2. **Older baseline path: rubric-based behavioral mining**

The mechanistic path is now the main project.

### Mechanistic path

The mechanistic path asks:

- given `chosen` vs `rejected` responses, what internal activation direction separates them?
- what text sequences move the model most along that direction?
- what sequences move against that direction?
- do these sequences eventually become semantically strange, repetitive, or opaque?

High-level flow:

1. author contrastive data in `data/axes/*.sample.jsonl`
2. extract activation-space direction vectors with `scripts/extract_mechanistic_directions.py`
3. mine sequences with `scripts/mine_pro_human_sequences.py`
4. merge shard outputs with `scripts/merge_mechanistic_sequences.py`
5. summarize results with `scripts/summarize_mechanistic_results.py`

### Baseline path

The baseline path asks:

- given a lexical rubric, what phrases make model outputs score higher on that rubric?

High-level flow:

1. define a lexical rubric in `data/direction_spec.sample.json`
2. validate it with `scripts/score_seed_pairs.py`
3. mine phrases with `scripts/mine_candidates.py`

This path is still useful for behavioral comparison, but it is no longer the main objective.

## 2. End-To-End Data And Control Flow

### 2.1 Data authoring flow

Files involved:

- `data/axes/*.sample.jsonl`
- `scripts/build_dataset_views.py`
- `data/seed_pairs.sample.jsonl`
- `data/eval_prompts.txt`

Flow:

1. Each axis file stores prompt-level `chosen` vs `rejected` comparisons.
2. `scripts/build_dataset_views.py` merges those files into:
   - `data/seed_pairs.sample.jsonl`
   - `data/eval_prompts.txt`

### 2.2 Mechanistic direction extraction flow

Files involved:

- `scripts/extract_mechanistic_directions.py`
- `src/humanity_direction/activations.py`
- `src/humanity_direction/pairs.py`
- `src/humanity_direction/prompting.py`
- `src/humanity_direction/scoring.py`

Flow:

1. Load `chosen` and `rejected` rows from `data/axes/`.
2. Build training-style prompt/completion text.
3. Extract hidden activations for chosen and rejected completions.
4. Compute `chosen - rejected` activation differences.
5. Average those differences:
   - per axis
   - globally across all axes
6. Save:
   - `directions.pt`
   - `summary.json`

### 2.3 Mechanistic sequence mining flow

Files involved:

- `scripts/mine_pro_human_sequences.py`
- `src/humanity_direction/activations.py`
- `src/humanity_direction/mechanistic_scoring.py`
- `src/humanity_direction/prompting.py`

Flow:

1. Load a saved direction vector from `directions.pt`.
2. Load a prompt pool, by default from `data/axes/`.
3. Build baseline prompt activations across that prompt pool.
4. Inject candidate sequences into each prompt in the pool.
5. Score each candidate by mean projection across prompts:
   - `score = mean(dot(h_candidate_prompt - h_base_prompt, unit_direction))`
6. Keep top candidates:
   - exhaustive search for shorter spaces
   - beam search for longer heuristic runs
7. Save retained candidate rows per shard.

### 2.4 Mechanistic merge and reporting flow

Files involved:

- `scripts/merge_mechanistic_sequences.py`
- `scripts/summarize_mechanistic_results.py`

Flow:

1. Merge shard-local top candidates into one global top set.
2. Write final `top_sentences.jsonl`.
3. Analyze score distributions, repetition, overlap, and motifs.
4. Write:
   - `report.json`
   - `report.md`

### 2.5 Rubric baseline flow

Files involved:

- `data/direction_spec.sample.json`
- `scripts/score_seed_pairs.py`
- `scripts/mine_candidates.py`
- `src/humanity_direction/direction.py`
- `src/humanity_direction/search.py`

Flow:

1. Score candidate output text against a lexical rubric.
2. Compare baseline vs injected completions.
3. Rank phrases by rubric score improvement.

This path evaluates external output behavior.
It does **not** directly optimize internal activation movement.

## 3. Root-Level Files

### `README.md`

Purpose:

- the main project README
- explains the current mechanistic objective
- documents the baseline path as secondary

Role in the project:

- first-stop orientation for humans
- explains the current research framing and main workflows

### `FILE_GUIDE.md`

Purpose:

- this file
- detailed per-file architecture map

Role in the project:

- long-form internal documentation
- useful for onboarding or refactoring

### `pyproject.toml`

Purpose:

- package metadata and Python dependency declaration

Key contents:

- package name: `humanity-direction`
- build backend: `setuptools`
- runtime dependencies:
  - `torch`
  - `transformers`
  - `safetensors`

Role in the project:

- makes the repo installable
- defines the package root at `src/`

### `update.sh`

Purpose:

- safe repo update helper for HPC login nodes

What it does:

- checks for a clean working tree
- fetches from the configured remote
- fast-forwards only
- updates submodules if present

Role in the project:

- safe "refresh code on cluster" helper
- avoids accidental updates over local edits

### `setup_hpc.sh`

Purpose:

- older `.venv`-based HPC environment setup

What it does:

- creates `.venv`
- installs the local package
- installs `transformers` and `safetensors`
- force-installs CUDA 12.1 `torch`

Role in the project:

- legacy environment path
- mostly replaced by the conda-based Explorer workflow

### `setup_hpc_conda.sh`

Purpose:

- current conda-based HPC environment setup

What it does:

- loads `explorer anaconda3/2024.06 cuda/12.1.1`
- recreates the `humanity-qwen25` environment
- installs the local package and model dependencies

Role in the project:

- the recommended Explorer runtime setup

### `fix_hpc_torch.sh`

Purpose:

- repair helper for broken or CPU-only Torch installs on HPC

What it does:

- reinstalls the package
- force-reinstalls CUDA 12.1 Torch
- verifies whether CUDA is visible

Role in the project:

- cluster troubleshooting tool

### `run_overnight_hpc_fairness.sh`

Purpose:

- old one-command launcher for rubric-based fairness mining using `.venv`

What it does:

- builds `.venv`
- submits a wrapped fairness mining GPU job

Role in the project:

- legacy convenience wrapper for the baseline path

### `run_overnight_hpc_fairness_conda.sh`

Purpose:

- conda-based launcher for the same fairness mining baseline

What it does:

- assumes the conda env already exists
- submits a fairness batch job

Role in the project:

- Explorer-compatible wrapper for the older baseline

### `run_all_axes_hpc_conda.sh`

Purpose:

- one-command batch launcher for the older all-axis rubric mining path

What it does:

- splits axes into predefined groups
- supports `MODE=parallel` or `MODE=series`
- submits `mine_all_axes.sbatch` multiple times with different axis subsets

Role in the project:

- batch orchestration for the older behavioral path

### `run_pro_human_mechanistic_hpc.sh`

Purpose:

- the main mechanistic HPC launcher

What it does:

- optionally runs direction extraction if `DIRECTION_TENSORS` is not supplied
- submits shard jobs for mechanistic sequence mining
- submits a merge job with dependencies
- supports both:
  - short exhaustive searches
  - long beam searches

Important environment knobs:

- `DIRECTION_TENSORS`
- `DIRECTION_NAME`
- `DIRECTION_SIGN`
- `SEARCH_MODE`
- `MIN_PHRASE_LEN`
- `MAX_PHRASE_LEN`
- `BEAM_WIDTH`
- `NUM_SHARDS`
- `BATCH_SIZE`

Role in the project:

- primary HPC orchestration entrypoint for the current research

### `report.md`

Purpose:

- ad hoc generated result summary

What it contains:

- a generated Markdown analysis artifact from a previous run
- the exact contents may change or be replaced as new reports are generated

Role in the project:

- analysis output, not source code
- not the canonical reporting tool
- superseded by `scripts/summarize_mechanistic_results.py`

## 4. Data Files

### `data/lexicon.txt`

Purpose:

- controlled search alphabet for sequence mining

Important detail:

- entries are **lexicon units**
- they are not raw tokenizer tokens
- some entries are multiword, for example:
  - `human dignity`
  - `protect agency`
  - `care with honesty`

Role in the project:

- search space for both:
  - baseline phrase mining
  - mechanistic sequence mining

### `data/direction_spec.sample.json`

Purpose:

- lexical rubric definition for the older baseline path

Structure:

- one named direction
- multiple axes
- each axis has:
  - `weight`
  - `positive_cues`
  - `negative_cues`

Role in the project:

- baseline output scoring only
- no longer the main optimization target

### `data/seed_pairs.sample.jsonl`

Purpose:

- merged seed-pair view

What it contains:

- rows pulled from all axis files
- each row keeps:
  - `axis`
  - `prompt`
  - `chosen`
  - `rejected`

Role in the project:

- convenient merged data artifact
- used for inspection and lightweight tooling

### `data/eval_prompts.txt`

Purpose:

- merged prompt-only view

What it contains:

- one prompt per line
- deduplicated across axis files

Role in the project:

- evaluation or probing prompt pool

### `data/axes/*.sample.jsonl`

These are the source datasets for the project.
All of them follow the same schema:

- `axis`
- `prompt`
- `chosen`
- `rejected`

Each file defines one behavioral sub-direction.

#### `data/axes/accountability.sample.jsonl`

Purpose:

- contrastive examples about owning mistakes, correcting errors, and not hiding problems

#### `data/axes/boundaries.sample.jsonl`

Purpose:

- contrastive examples about limits, response expectations, availability, and role boundaries

#### `data/axes/conflict_resolution.sample.jsonl`

Purpose:

- examples about de-escalation, structured disagreement, repair, and resolution

#### `data/axes/empathy.sample.jsonl`

Purpose:

- examples about support, emotional sensitivity, non-shaming responses, and compassionate handling

#### `data/axes/fairness.sample.jsonl`

Purpose:

- examples about equal treatment, consistent process, proper attribution, and anti-favoritism

#### `data/axes/feedback.sample.jsonl`

Purpose:

- examples about constructive feedback, specificity, and improvement-oriented communication

#### `data/axes/inclusion.sample.jsonl`

Purpose:

- examples about accessible participation, anti-exclusion, respectful naming, and broader input

#### `data/axes/integrity.sample.jsonl`

Purpose:

- examples about honesty, accurate reporting, resisting misrepresentation, and preserving records

#### `data/axes/leadership.sample.jsonl`

Purpose:

- examples about responsible direction-setting, clarity, fairness in decision-making, and accountable leadership

#### `data/axes/learning.sample.jsonl`

Purpose:

- examples about reflection, improvement, openness to learning, and non-defensive adaptation

#### `data/axes/ownership.sample.jsonl`

Purpose:

- examples about initiative, follow-through, fixing handoffs, and staying engaged until resolution

#### `data/axes/privacy.sample.jsonl`

Purpose:

- examples about data minimization, access boundaries, confidentiality, and responsible handling of personal information

#### `data/axes/respect.sample.jsonl`

Purpose:

- examples about calm communication, equal voice, private correction, and non-contemptuous treatment

#### `data/axes/safety.sample.jsonl`

Purpose:

- examples about hazard reporting, respecting safety procedures, and not normalizing risky shortcuts

#### `data/axes/trust.sample.jsonl`

Purpose:

- examples about reliability, transparency, consent, and boundary-aware confidentiality

## 5. Python Package Modules (`src/humanity_direction/`)

### `src/humanity_direction/__init__.py`

Purpose:

- package export surface

What it does:

- re-exports the small baseline-oriented utility set:
  - `MiningConfig`
  - rubric scoring types
  - pair loading helpers
  - beam search helpers

Role in the project:

- convenience import layer
- still reflects the older baseline-centric package surface more than the new mechanistic path

### `src/humanity_direction/config.py`

Purpose:

- config dataclass for the older baseline miner

Key object:

- `MiningConfig`

Fields cover:

- model name
- rubric file
- lexicon file
- output paths
- prompt/pairs inputs
- axis filters
- beam width
- max phrase length
- top-k
- generation parameters

Role in the project:

- used by `scripts/mine_candidates.py`

### `src/humanity_direction/data.py`

Purpose:

- generic data I/O helpers

Functions:

- `load_jsonl(path)`
  - loads JSON Lines into a list of dicts
- `load_lines(path)`
  - loads non-empty lines from a plain text file
- `write_jsonl(path, rows)`
  - writes iterable rows as JSONL
- `write_lines(path, lines)`
  - writes plain text lines

Role in the project:

- shared file I/O utility used across scripts

### `src/humanity_direction/direction.py`

Purpose:

- lexical-rubric loading and scoring

Key dataclasses:

- `DirectionAxis`
- `DirectionSpec`
- `AxisScore`
- `DirectionScore`

Key functions:

- `_normalize(text)`
  - lowercases and normalizes whitespace for substring matching
- `load_direction_spec(path)`
  - parses the lexical rubric JSON
- `score_text_against_direction(text, direction)`
  - counts positive and negative cue matches and returns weighted axis scores

Role in the project:

- core of the older baseline path
- still useful for behavioral validation, but not the main mechanistic score

### `src/humanity_direction/pairs.py`

Purpose:

- pair-loading helpers

Functions:

- `load_pairs(path)`
  - loads a single JSONL file or merges all `*.jsonl` files from a directory
- `collect_prompts(rows)`
  - deduplicates and extracts prompt text from pair rows

Role in the project:

- common data entry layer for:
  - baseline scripts
  - mechanistic scripts

### `src/humanity_direction/prompting.py`

Purpose:

- prompt construction utilities

Constants:

- `SYSTEM_PROMPT`
  - default built-in pro-human system text used across prompt builders
  - this matters because sequence mining and direction extraction are measured relative to prompts that already contain a pro-human prior

Functions:

- `build_chat_prompt(user_prompt, system_prompt=SYSTEM_PROMPT)`
  - builds a chat-format prompt string
- `build_injected_prompt(user_prompt, phrase)`
  - injects a candidate steering phrase into the system message
- `build_training_example(prompt, completion)`
  - returns:
    - prompt prefix
    - prompt + completion full text

Role in the project:

- shared prompt-template layer for both baseline and mechanistic runs
- not neutral infrastructure: it injects a default normative framing before any additional steering phrase is added

### `src/humanity_direction/scoring.py`

Purpose:

- model execution helpers

Functions:

- `choose_device()`
  - selects `cuda`, then `mps`, otherwise `cpu`
- `normalized_logprob(...)`
  - scores target text under a prompt
  - currently not central to the current mechanistic path
- `generate_completion(...)`
  - generates model text from a prompt

Role in the project:

- generation backbone for the older baseline path
- also used by `scripts/mine_mechanistic_dataset.py`

### `src/humanity_direction/search.py`

Purpose:

- generic phrase beam search utilities

Key dataclass:

- `CandidateResult`

Functions:

- `_dedupe_preserve_order(items)`
  - removes duplicates while keeping order
- `beam_search_phrases(...)`
  - scores seed terms
  - keeps a beam
  - expands phrases level by level
  - can optionally return only top-K

Role in the project:

- used by:
  - `scripts/mine_candidates.py`
  - `scripts/mine_mechanistic_dataset.py`

### `src/humanity_direction/activations.py`

Purpose:

- hidden-state extraction utilities for the mechanistic path

Functions:

- `load_model_and_tokenizer(model_name, device)`
  - loads a causal LM and tokenizer
  - sets pad token if missing
  - chooses half precision on CUDA
- `batch_terminal_activations(...)`
  - runs a batch of texts
  - extracts the hidden state at the final attended token position
- `mean_completion_activation(...)`
  - extracts the hidden states for completion tokens only
  - averages them into one vector

Role in the project:

- the core mechanistic representation-extraction layer

Important distinction:

- `batch_terminal_activations` is used by the new sequence miner
- `mean_completion_activation` is used by direction extraction and the older mechanistic dataset miner

### `src/humanity_direction/mechanistic_scoring.py`

Purpose:

- activation-space direction loading and projection scoring

Functions:

- `load_direction_vector(path, direction_name="global")`
  - loads `directions.pt`
  - returns:
    - a unit vector
    - metadata such as model name, layer, and direction name
- `projection_score(delta, unit_direction)`
  - computes a dot product score

Role in the project:

- the simplest possible mechanistic scoring core

## 6. Python Scripts (`scripts/`)

### `scripts/build_dataset_views.py`

Purpose:

- regenerate merged dataset artifacts from `data/axes/`

Main flow:

1. load all rows via `load_pairs`
2. collect unique prompts
3. write merged JSONL
4. write prompt list

Role in the project:

- data maintenance utility

### `scripts/score_seed_pairs.py`

Purpose:

- validate whether the lexical rubric prefers the `chosen` side of each pair

Main flow:

1. load the rubric
2. load all pairs
3. score `chosen`
4. score `rejected`
5. print per-row deltas and total win count

Role in the project:

- baseline sanity check
- useful when debugging `data/direction_spec.sample.json`

### `scripts/mine_candidates.py`

Purpose:

- old baseline phrase miner based on rubric improvement

Key helper:

- `write_axis_split_outputs(...)`
  - if a run mixes multiple axes, split the saved outputs into `by_axis/`

Main flow:

1. parse a `MiningConfig`
2. load rubric, prompts, and seed terms
3. load the model
4. generate baseline completions for prompts
5. define `score_phrase(phrase)` as average rubric improvement
6. run `beam_search_phrases(...)`
7. rescore top phrases with full per-prompt detail
8. write:
   - candidate ranking
   - dataset rows
   - optional per-axis splits

Role in the project:

- main implementation of the old behavioral/rubric path

### `scripts/mine_mechanistic_dataset.py`

Purpose:

- earlier mechanistic miner that scores phrases by internal projection but still depends on generated completions

Main flow:

1. load prompts and direction vector
2. generate baseline completions
3. extract mean completion activations
4. define `score_phrase(phrase)` by average projection shift
5. run `beam_search_phrases(...)`
6. rescore top phrases with full per-prompt detail
7. write:
   - `top_sentences.jsonl`
   - `dataset.jsonl`
   - `summary.json`

Role in the project:

- intermediate mechanistic approach
- conceptually useful
- less central than the newer direct sequence miner

### `scripts/extract_mechanistic_directions.py`

Purpose:

- build the activation-space direction vectors

Main flow:

1. load pairs and filter by axis if requested
2. group rows by axis
3. load the model
4. for each row:
   - build prompt/completion text
   - extract `chosen` activation
   - extract `rejected` activation
   - compute difference
5. average the differences
6. compute diagnostics:
   - vector norm
   - projection stats
   - cosine coherence
7. save:
   - `directions.pt`
   - `summary.json`

Role in the project:

- first mechanistic step
- defines the target direction for all later mining

### `scripts/mine_pro_human_sequences.py`

Purpose:

- current main mechanistic sequence miner

This is the most important current script.

Config fields:

- model and direction paths
- `direction_sign`
- `search_mode`
- phrase length limits
- `batch_size`
- `retain_top_k`
- `beam_width`
- shard information

Key helpers:

- `sequence_generator(...)`
  - exhaustive generator over lexicon-unit sequences
- `batched(...)`
  - generic batching helper
- `push_candidate(...)`
  - keeps only the highest-scoring retained rows
- `score_phrase_batch(...)`
  - batch-scores candidate phrases by activation projection
- `log_progress(...)`
  - prints search progress

Main flow:

1. load and normalize the direction vector
2. optionally flip sign with `direction_sign`
3. load prompt pool, lexicon, and model
4. compute baseline activations for the prompt pool
5. branch by search mode:

   - `exhaustive`
     - enumerate all lexicon-unit sequences in a shard
     - score candidates by mean projection across prompts

   - `beam`
     - score length-1 units
     - keep the top frontier
     - expand frontier step by step
     - score expansions by mean projection across prompts

6. retain the top-scoring rows
7. save:
   - `retained_candidates.jsonl`
   - `summary.json`

Role in the project:

- current main search engine for mechanistic dataset construction

### `scripts/merge_mechanistic_sequences.py`

Purpose:

- merge shard-local candidate sets into one global top-K

Main flow:

1. find `retained_candidates.jsonl` under each shard
2. deduplicate by `steering_sentence`
3. keep the best score for duplicates
4. sort globally
5. clip to final top-K
6. assign `global_rank`
7. save:
   - `top_sentences.jsonl`
   - `summary.json`

Role in the project:

- final aggregation step after sharded mining

### `scripts/summarize_mechanistic_results.py`

Purpose:

- rich analysis and reporting for mechanistic runs

Important internal structures:

- `RunData`
  - bundles rows, root path, summary data, and reconstructed unit segmentation

Key helpers:

- `load_jsonl(...)`
- `load_lines(...)`
- `percentile(...)`
- `numeric_stats(...)`
- `top_slice_mean(...)`
- `default_output_dir(...)`
- `load_summary(...)`
- `load_run(...)`
- `segment_sequence(...)`
  - reconstructs a saved sequence back into lexicon units
- `summarize_run(...)`
  - computes within-run summary metrics
- `compare_runs(...)`
  - computes overlap and distinctive units between runs
- `render_markdown(...)`
  - turns the report into human-readable Markdown

Main flow:

1. load one or more run roots
2. segment saved sequences back into lexicon units
3. compute:
   - score statistics
   - delta-norm statistics
   - length profile
   - repetition metrics
   - unit, bigram, trigram frequencies
   - top examples
4. if multiple runs are given:
   - compute overlap
   - compute unit-level differences
5. write:
   - `report.json`
   - `report.md`
6. also print the Markdown summary to stdout

Role in the project:

- the main analysis/reporting tool for mechanistic results

## 7. Slurm Entry Points (`sbatch/`)

### `sbatch/setup_env.sbatch`

Purpose:

- old batch wrapper around `setup_hpc.sh`

Role:

- legacy `.venv` environment setup

### `sbatch/score_pairs.sbatch`

Purpose:

- batch wrapper around `scripts/score_seed_pairs.py`

Role:

- rubric validation job

### `sbatch/mine_fairness.sbatch`

Purpose:

- older single-axis rubric mining job

What it does:

- activates either conda or `.venv`
- checks CUDA
- runs `scripts/mine_candidates.py` restricted to one axis

Role:

- fairness-focused behavioral baseline

### `sbatch/mine_all_axes.sbatch`

Purpose:

- older multi-axis rubric mining job

What it does:

- activates runtime
- accepts optional `AXES`
- runs `scripts/mine_candidates.py`

Role:

- baseline multi-axis mining

### `sbatch/mine_mechanistic_dataset.sbatch`

Purpose:

- batch wrapper for the older mechanistic dataset miner

What it does:

- activates runtime
- loads `DIRECTION_TENSORS`
- optionally filters by axis
- runs `scripts/mine_mechanistic_dataset.py`

Role:

- earlier mechanistic batch path
- less central than the newer shard + merge pipeline

### `sbatch/extract_mechanistic_directions.sbatch`

Purpose:

- batch wrapper for direction extraction

What it does:

- activates runtime
- checks CUDA
- runs `scripts/extract_mechanistic_directions.py`

Role:

- standard HPC entrypoint for creating `directions.pt`

### `sbatch/mine_pro_human_sequences_shard.sbatch`

Purpose:

- batch wrapper for one mechanistic mining shard

What it does:

- activates runtime
- checks CUDA
- passes direction, sign, search mode, and shard parameters into `scripts/mine_pro_human_sequences.py`

Role:

- primary compute job for the current mechanistic path

### `sbatch/merge_mechanistic_sequences.sbatch`

Purpose:

- merge wrapper for mechanistic shard outputs

What it does:

- activates conda
- runs `scripts/merge_mechanistic_sequences.py`

Important operational note:

- it still requests a GPU on Explorer because the public partition rejects jobs that do not match the queue policy
- the merge itself is CPU-light

Role:

- final aggregation step in the HPC pipeline

### `sbatch/README.md`

Purpose:

- dedicated documentation for the Slurm workflow

Role:

- operational reference for Explorer usage

## 8. How The Pieces Fit Together

### If you want to change the behavioral concept itself

Edit:

- `data/axes/*.sample.jsonl`

Then rebuild:

- `scripts/build_dataset_views.py`

If you still care about the old lexical baseline, also inspect:

- `data/direction_spec.sample.json`

### If you want to change the internal direction being extracted

Edit or vary:

- `scripts/extract_mechanistic_directions.py`
- `src/humanity_direction/activations.py`
- `src/humanity_direction/prompting.py`

Typical knobs:

- layer index
- which axes are included
- which activation readout is used

### If you want to change what counts as a strong mechanistic mover

Edit:

- `src/humanity_direction/mechanistic_scoring.py`
- `scripts/mine_pro_human_sequences.py`

Typical knobs:

- projection sign
- prompt pool source
- search mode
- beam width
- retention policy

### If you want to change HPC behavior

Edit:

- `run_pro_human_mechanistic_hpc.sh`
- `sbatch/extract_mechanistic_directions.sbatch`
- `sbatch/mine_pro_human_sequences_shard.sbatch`
- `sbatch/merge_mechanistic_sequences.sbatch`

Typical knobs:

- number of shards
- time limits
- memory
- batch size
- partition use

### If you want to understand or compare results

Use:

- `scripts/summarize_mechanistic_results.py`

That is the main reporting layer for:

- collapse/repetition
- motif frequency
- run overlap
- distinctive unit usage

## 9. Practical Reading Order

If someone new joins the project, the fastest correct reading order is:

1. `README.md`
2. `data/axes/*.sample.jsonl`
3. `scripts/extract_mechanistic_directions.py`
4. `src/humanity_direction/activations.py`
5. `src/humanity_direction/mechanistic_scoring.py`
6. `scripts/mine_pro_human_sequences.py`
7. `run_pro_human_mechanistic_hpc.sh`
8. `scripts/summarize_mechanistic_results.py`
9. `sbatch/README.md`

If someone wants the older baseline path too, then read:

10. `data/direction_spec.sample.json`
11. `src/humanity_direction/direction.py`
12. `scripts/mine_candidates.py`
13. `run_all_axes_hpc_conda.sh`
