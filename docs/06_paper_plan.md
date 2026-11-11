# Paper plan

**Title options:**
1. What Abstention Deletes: Anti-Hallucination Training Silences Rare Knowledge
2. The Tail Tax: Who Pays When Language Models Learn to Say "I Don't Know"

## Structure (8 pages)

1. **Introduction**, with Figure 1: Gain and Loss bars per frequency bucket for each method. This is the tail going silent.
2. **Setup:** exact counts, the identity, the tail tax, matched coverage.
3. **G0 and E2:** the audit across methods.
4. **E3:** is training smarter than a threshold?
5. **E4:** the singleton bound with exact counts.
6. **E5:** a fix that keeps the tail.
7. **Related work, limitations, ethics.**

## Headline (fill from results only)

"Across [N] anti-hallucination methods, [X]% of the correct answers given up are facts seen at most twice in
pretraining, [TT]× their share. At matched coverage, [M of N] methods do no better than thresholding the base
model's confidence."

## Reviewer objections

1. **"Counts are a noisy proxy."**
   - We condition on base correctness.
   - We use two count definitions and a window sensitivity check.
2. **"Only OLMo."**
   - Exploration X1 replicates on released checkpoints with popularity proxies.
3. **"Obvious."**
   - The matched-coverage comparison and the exact singleton test are not obvious.
   - A tail tax near 1 would also have been reported.
