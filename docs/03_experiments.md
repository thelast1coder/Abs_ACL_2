# Experiments

Status codes: READY, TODO, RUNNING, DONE.

## D0: data (laptop). READY

1. Load PopQA: `abst.facts`.
2. Get exact counts for every fact: `abst.counts`.
3. Draw a bucket-balanced sample, up to 1,000 per bucket: `abst.sample`. The sample excludes few-shot subjects.

## G0: tail tax of the shipped instruct model. READY (about 3 GPU-hours)

- **Models.** OLMo-2-1124-7B (base) and OLMo-2-1124-7B-Instruct.
- **Base prompt.** 5-shot QA, greedy, 16 tokens, stop at newline. Plus 10 pure samples per fact for confidence.
- **Instruct variants.** S1 "Answer briefly." and S2 "... If you are not sure, say I don't know". Greedy, 48 tokens.
- **Statistics.** Per-bucket table, TT with a bootstrap CI, and matched-coverage TT. The gate is in `docs/04`.

## E1: more tail facts and count robustness. TODO (0.5 GPU-h)

- **More facts.** Add EntityQuestions and TailCat (from MB5) if the tail buckets are thin.
- **Count variants.** Window 1000; subject aliases; exact sentence patterns for templated relations.
- **Pass condition.** Conclusions must hold under every count variant.

## E2: method audit. TODO (5 GPU-h)

| Method | How | Model | GPU-h |
|---|---|---|---|
| R-Tuning style refusal-aware SFT | LoRA on targets built from B's correctness | OLMo-2-7B | 1.5 |
| GRPO abstention (KARL/TruthRL style) | reward +1 correct, 0 abstain, −1 wrong | OLMo-2-1B (base and SFT) | 2.0 |
| Prompted abstention | S2 | 7B | (G0) |
| Semantic-entropy threshold | 10 samples, cluster, abstain above a threshold | 7B | 1.0 |
| Released checkpoints (X1) | KARL / TruthRL / Abstain-R1 if public; popularity proxy | Llama/Qwen | 1.5 (exploration) |

## E3: matched-coverage frontiers. TODO (CPU)

- For every method: per-bucket loss against the coverage-matched base threshold.
- Also the full (coverage, hallucination) curve per bucket.

## E4: singleton bound. TODO (0.5 GPU-h)

- Per-relation s_r, h_r and ECE for the base model.
- Then where the abstentions fall.

## E5: fix. TODO (2 GPU-h)

- **Targets.** Consistency-based abstention targets, then LoRA.
- **Metric.** Tail loss recovered at equal hallucination.
- **Gate.** G2 requires at least 30% recovery at no more than +1 point of hallucination.
