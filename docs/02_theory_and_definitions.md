# Definitions and maths (the code must match this file)

## Objects

- **Fact f.** A subject, a relation and a set of accepted answers (PopQA `possible_answers`).
- **Exposure c(f).** The infini-gram CNF count of `subject AND (answer1 OR answer2 ...)` within 100 tokens on `v4_olmo-mix-1124_llama`.
  - It is a lower bound.
  - Buckets k: 0, 1, 2, 3-5, 6-20, 21-100, >100.
  - The tail is k ∈ {1, 2}. Facts with c = 0 are reported but not in the tail: zero exposure means the base model cannot know them except by inference.
- **Models.** B is the base model. S is a model trained, or prompted, to abstain.
  - G0 uses two S variants: S1 = OLMo-2-Instruct as shipped; S2 = OLMo-2-Instruct told to say "I don't know" when unsure.
- **correct(r).** An accepted answer appears in the normalised response (the PopQA metric).
- **abstained(r).** The response matches the abstention regex and contains no accepted answer. A hedged correct answer counts as correct.

## Exact bookkeeping (an identity)

Per bucket k:
- C_B(k) = #B correct;
- C_S(k) = #S correct;
- Gain(k) = #(B wrong ∧ S correct);
- Loss(k) = #(B correct ∧ S not correct) = Loss_abs + Loss_flip.

Then **C_S(k) − C_B(k) = Gain(k) − Loss(k)** exactly. The code asserts it.

## Tail tax

TT = [Loss(tail)/Loss(all)] ÷ [C_B(tail)/C_B(all)]

- **TT = 1:** the tail loses in proportion to what it contributes.
- **TT = 2:** the tail pays double.
- **CI:** nonparametric bootstrap over facts (1,000 resamples).

## Matched-coverage reference

- **Base confidence:** conf(f) = the share of 10 pure samples (T = 1, top_p = 1, top_k = 0) that agree with B's greedy answer.
- **Threshold policy:** π_τ answers if conf ≥ τ. Pick τ so that π_τ's coverage equals S's coverage.
- **Comparison:** compute the threshold policy's losses and TT.
  - If S's tail losses are no smaller than π_τ's, training learned nothing about the tail that a confidence threshold would not do.

## Singleton bound test (E4; Kalai, Nachum, Vempala, Zhang 2025)

For each relation r:
- the singleton rate s_r is the share of facts with c(f) = 1;
- h_r is the base model's hallucination rate (answered and wrong);
- the calibration error is ECE of the base model's confidence.

Test h_r ≥ s_r − calibration error. Then show where each S's abstentions fall by bucket.

**Ownership.** The surprise-weighted refinement of this bound belongs to SNR. Do not duplicate it here; cite it.

## Fix (E5)

- **Targets.** Build abstention targets from B's *consistency* (samples disagree → abstain), not from B's *correctness*.
- **Hypothesis.** This keeps rare-but-known facts, which B answers consistently even at low familiarity.
- **Report.** Tail loss recovered, at an equal overall hallucination rate.
