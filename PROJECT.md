# PROJECT.md: ABS, "What Abstention Deletes" (ACL 2027)

Single source of context for humans and agents. Read it fully before acting. Details live in `docs/`.

- Venue: ACL 2027 via ARR January cycle. **Deadline 4 January 2027, 23:59 AoE.** Long paper.
- Target area: **Special Theme (Homogenization and knowledge collapse)**. Target: oral; Outstanding or Theme award.
- Owner: Dawar (PI), Erdős AI Lab. Engineering: Claude Code. Compute: one 80 GB GPU VM.
- Budget: **about 12 GPU-hours in total**. G0 costs about 3.

---

## 1. The idea in plain language

**What the field does.** The standard fix for hallucination today is to teach models to say
"I don't know" (abstention training, refusal-aware tuning, RL rewards for honesty, confidence
thresholds). Papers report two numbers: hallucinations fall a lot, and accuracy falls a little.

**What nobody asks.** Which correct answers did the model stop giving?

**Our bet.** They are mostly facts the model saw only once or twice in pretraining. A model can know
a rare fact and still feel unsure about it, so a model trained to abstain when unsure goes quiet on
exactly the rare facts it knew. "Safer" models drift towards popular knowledge, and the usual
overall metrics hide it. That is knowledge collapse, caused by safety training.

**How we show it.** Exact bookkeeping on an open model whose pretraining data we can count
(OLMo-2 + infini-gram):
- per fact, did the base model know it, and did the safety-trained model keep it?
- grouped by how many times the fact appeared in pretraining.

## 2. Problem, causes, attack

**Open problem.** Every abstention method (KARL, TruthRL, Abstain-R1, R-Tuning, GRAIT, I-CALM, ...)
reports aggregate accuracy and abstention. None reports coverage by pretraining frequency. So the field
cannot tell a method that removes guesses from one that removes knowledge.

**Causes (hypothesis).**
- **Confidence tracks familiarity, which tracks frequency.** Rare-but-known facts sit at low confidence.
- **Training uses confidence or correctness as the abstain signal.** Abstention training then cuts at a
  confidence threshold, which lands on rare facts.
- **The theory points the same way.** Kalai et al. (2025) show that facts seen once in training
  (singletons) are where a calibrated model must hallucinate. Abstention "solves" this by silencing
  singletons, including the ones the model knew.

**Attack.**
1. Exact counts c(f).
2. A bookkeeping identity per frequency bucket.
3. A tail-tax statistic.
4. A matched-coverage baseline.
5. An audit of several methods.
6. A test of the singleton bound.
7. A cheap fix.

## 3. Claims and pre-registered predictions

| # | Claim | Kill / weaken if |
|---|---|---|
| A1 | Losses concentrate in the tail: tail tax TT ≥ 1.5 for the shipped instruct model | TT < 1.2 for every variant (G0 → FOLD into MB5) |
| A2 | TT ≥ 1.5 for at least 2 of 4 abstention methods | fewer than 2 (weaken to "some methods") |
| A3 | At matched coverage, trained methods are no better than thresholding the base model's confidence in the tail | trained methods clearly beat the threshold (then that is the finding) |
| A4 | The singleton bound holds with exact counts, and abstentions fall on singletons | bound violated or abstentions spread evenly |
| A5 | Consistency-based abstention targets recover ≥ 30% of tail loss at ≤ +1 point hallucination | recovery < 15% (drop the fix, keep the audit) |

**Identity (exact; checked in code by an assert).** For a frequency bucket k, with B the base model
and S the abstention-trained model:

C_S(k) − C_B(k) = Gain_S(k) − Loss_S(k)

**Tail tax.**

TT = [Loss(k≤2)/Loss(all)] ÷ [C_B(k≤2)/C_B(all)]

## 4. Goals and success levels

- **Findings-level:** A1 on OLMo-2 plus one more method.
- **Oral-level:**
  - A1 and A2;
  - A3 with clean matched-coverage curves;
  - A4.
- **Award-level:** all of the above, plus A5 (a fix that keeps the tail). This becomes the new column every
  abstention paper reports: coverage by frequency.

## 5. Constraints

1. **Compute.** About 12 GPU-hours, logged in `logs/gpu_ledger.csv`.
   - If the GPU is shared, never kill others' processes; keep 8 GiB headroom.
2. **Deadline.** 4 January 2027 AoE; writing freeze 20 December.
3. **Counts.**
   - Use `v4_olmo-mix-1124_llama` (OLMo-2 pretraining). Counts are lower bounds.
   - Dolmino mid-training is not indexed: state this as a limitation.
4. **No invented numbers.**
5. **Sibling boundaries.**
   - SNR owns the surprise-weighted floor and type priors. MB5 owns sampler truncation.
   - This paper owns abstention-training effects and the raw singleton test.
   - MB5, SNR and ABS share facts and counts infrastructure. Cite them as anonymous concurrent work, and say in one sentence how each differs.
6. **Style.** British spelling, no em dashes, plain English. Maths only where it defines a measured quantity.

## 6. Operating model

- **Roles.** The PI signs gates. The agent builds, tests, runs and reports. A red-team reviewer checks each gate report.
- **Gate reports** go in `results/gates/G*.md`. Report in this order:
  1. the verdict;
  2. the deciding numbers with CIs;
  3. the validity checks;
  4. GPU-hours spent;
  5. anything surprising;
  6. the next step.
- **Decisions.** Every decision goes in `notes/decisions.md`. Threshold changes after seeing data need a logged reason and a line in the paper.
- **Discipline.**
  - Tests come first.
  - Each run writes `meta.json`.
  - Seeds are fixed.
  - The abstention regex is validated by hand on 200 responses before G0 is signed.

## 7. Exploitation and exploration

About 80% of GPU-hours go to the core: G0, E1-E4. About 20% go to capped exploration.

| Item | What | Cap | Promote if |
|---|---|---|---|
| X1 | Released abstention checkpoints (KARL, TruthRL, Abstain-R1) on Llama/Qwen, with Wikipedia popularity as the count proxy | 1.5 GPU-h | same TT direction |
| X2 | Second count definition: exact sentence-pattern counts for templated relations, and window 1000 | 0 GPU (CPU) | conclusions stable |
| X3 | Scale: OLMo-2 13B pair | 1.0 GPU-h | TT holds |

**Rules.**
- Exploration never blocks a gate.
- If the core is late, exploration stops first.
- Promotion is a PI decision, and it is logged.

## 8. Experiment map and dependencies (details in `docs/03_experiments.md`)

| ID | Question | Needs | GPU-h |
|---|---|---|---|
| D0 | PopQA + exact counts + bucket-balanced sample (laptop) | none | 0 |
| G0 | Does the shipped instruct model's loss concentrate in the tail? | D0 | 3.0 |
| E1 | More tail facts (EntityQuestions, MB5 TailCat); count sensitivity | G0 | 0.5 |
| E2 | Method audit: R-Tuning LoRA (7B), GRPO abstention (1B), prompted and semantic-entropy abstention | G0 = GO | 5.0 |
| E3 | Matched-coverage frontiers per bucket for every method | E2 | 0.5 (CPU mostly) |
| E4 | Singleton bound test (Kalai) with exact counts | D0, E1 | 0.5 |
| E5 | Fix: consistency-based abstention targets | E2 | 2.0 |
| R | Reserve | | 0.5 |

```
D0 ──> G0 ──GO──> E1 ──> E4
              └──> E2 ──> E3
                    └──> E5 (fix)        G1 after E2: ≥ 2 methods with TT ≥ 1.5
                                         G2 after E5: recovery ≥ 30% at ≤ +1 pt hallucination
```

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tail facts too few or unknown to the base model | Medium | High | Add EntityQuestions/TailCat (E1); G0 V1/V2 checks |
| Co-occurrence count ≠ "knowing" | High | Medium | Condition on base correctness; two count definitions; window sensitivity |
| Abstention regex misclassifies | Medium | Medium | Hand-validate 200 responses; report agreement |
| "Obvious" objection | Medium | Medium | The matched-coverage comparison and exact singleton test are not obvious; a null TT is reported honestly |
| Collision with an abstention paper adding a frequency split | Medium | High | Weekly arXiv sweep; post early if needed |

## 10. Status and next actions

- **2026-11-11.** G0 code is written and tested on CPU (6 tests). No data built yet: this container cannot reach Hugging Face or infini-gram.
- **Next, laptop:** run `bash scripts/run_g0.sh laptop`. This downloads PopQA, gets about 14k exact counts (takes a while; it is cached and resumable) and draws the sample.
- **Next, GPU:**
  1. `SMOKE=1 bash scripts/run_g0.sh gpu`, then read 20 instruct responses by eye.
  2. `bash scripts/run_g0.sh gpu`
  3. `bash scripts/run_g0.sh analyse`
