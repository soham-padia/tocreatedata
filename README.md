# Humanity Direction Research Scaffold

This project turns your idea into a concrete baseline:

1. Define a direction as a weighted rubric for "pro-human / positive humanity".
2. Start with seed preference pairs that clarify the direction.
3. Search for words or short phrases that make the base model generate outputs that score higher on that rubric.
4. Save the highest-impact phrases and prompt/completion examples as the next-round dataset.

The core point is to operationalize the direction. Do not treat "positive humanity" as one vague label. Break it into behaviors such as:

- protect human welfare
- preserve dignity and agency
- favor cooperation over domination
- be honest about uncertainty
- avoid dehumanization and exploitation

## Project Layout

- `data/axes/*.sample.jsonl`: starter preference pairs grouped by axis
- `data/seed_pairs.sample.jsonl`: merged starter preference pairs
- `data/direction_spec.sample.json`: weighted rubric for the target direction
- `data/eval_prompts.txt`: prompts used to probe the direction
- `data/lexicon.txt`: seed words/phrases for mining
- `scripts/build_dataset_views.py`: rebuilds merged pair and prompt views from `data/axes/`
- `scripts/score_seed_pairs.py`: checks whether your seed pairs match the direction rubric
- `scripts/mine_candidates.py`: mines phrases that push the base model toward the direction rubric
- `src/humanity_direction/`: reusable scoring, search, and prompt utilities

## Recommended Research Loop

### Stage 1: Define the direction

Create a direction rubric that breaks "positive humanity" into measurable axes such as:

- human dignity
- human agency
- cooperation
- truthfulness
- non-exploitation

The sample format is in `data/direction_spec.sample.json`.

Then author many seed pairs per axis with:

- `prompt`
- `chosen`
- `rejected`
- required `axis`
- optional `notes`

The `chosen` answer should be more human-protective, cooperative, and reality-grounded than `rejected`.
One axis should contain many situations, not one situation.
`data/axes/fairness.sample.jsonl` shows the intended pattern.

If you update axis files, rebuild the merged views:

```bash
python3 scripts/build_dataset_views.py \
  --pairs-path data/axes \
  --seed-output data/seed_pairs.sample.jsonl \
  --prompts-output data/eval_prompts.txt
```

### Stage 2: Validate the direction

Before mining, make sure your rubric roughly agrees with your own chosen/rejected examples:

```bash
python3 scripts/score_seed_pairs.py \
  --direction-file data/direction_spec.sample.json \
  --pairs-path data/axes
```

If the rubric does not consistently prefer `chosen` over `rejected`, fix the direction spec first.

### Stage 3: Mine effective words and phrases

For each evaluation prompt:

1. Generate a baseline completion from the base model.
2. Generate an injected completion from the base model with a candidate phrase added to the system prompt.
3. Score both completions against the direction rubric.
4. Keep phrases whose injected completion improves the direction score.
5. Expand the best phrases by beam search.

Run:

```bash
python3 scripts/mine_candidates.py \
  --model-name "YOUR_QWEN_CHECKPOINT" \
  --direction-file data/direction_spec.sample.json \
  --pairs-path data/axes \
  --lexicon-file data/lexicon.txt \
  --output-file outputs/mined_candidates.jsonl \
  --dataset-file outputs/mined_dataset.jsonl
```

To mine only one axis such as fairness:

```bash
python3 scripts/mine_candidates.py \
  --model-name "YOUR_QWEN_CHECKPOINT" \
  --direction-file data/direction_spec.sample.json \
  --pairs-path data/axes \
  --axis fairness \
  --lexicon-file data/lexicon.txt \
  --output-file outputs/fairness_candidates.jsonl \
  --dataset-file outputs/fairness_dataset.jsonl
```

## What The Mining Script Actually Measures

For a prompt `p` and candidate phrase `c`, the score is:

`score(c) = direction_score(base(inject(c, p))) - direction_score(base(p))`

If the score is positive, the phrase moves the model's own completion toward the direction. That gives you a direct trial-and-error signal without any extra training step.

## Why This Baseline Is Useful

- It treats the direction as the primary object, not a derived model checkpoint.
- It gives you a measurable objective for trial-and-error search.
- It creates a growing mined dataset that can be reused in later rounds.

## Known Limits

- The rubric is only a proxy for the direction you really care about.
- Phrase effects may overfit to the prompt template.
- Positive rubric shift does not prove robust value change.
- This is a research scaffold, not a safety guarantee.

## Next Experiments Worth Trying

- replace lexical rubric scoring with an LLM judge or pairwise preference judge
- swap beam search for an evolutionary search over phrases
- cluster successful phrases by semantic axis
- compare prompt steering against activation steering
- train round `n + 1` on mined examples and measure whether the mined phrases become unnecessary
