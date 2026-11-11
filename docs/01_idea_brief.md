# One-page brief: What Abstention Deletes

**Problem.** Anti-hallucination methods teach models to say "I don't know". They report that hallucinations fall
a lot and accuracy falls a little, but never say *which* correct answers disappear.

**Bet.** The lost answers are mostly facts seen once or twice in pretraining. "Safer" models quietly forget the long
tail: knowledge collapse caused by safety training.

**Method.**
- Count every fact's exposure exactly in OLMo-2's pretraining data.
- Do exact Gain/Loss bookkeeping per frequency bucket.
- Compute a tail-tax statistic.
- Compare against a confidence-threshold baseline at matched coverage.
- Audit four abstention methods.
- Test Kalai et al.'s singleton bound with real counts.
- Try a fix that keeps the tail.

**Cost.** About 12 GPU-hours. G0 costs about 3.

**Status.** G0 code is ready and tested on CPU.
