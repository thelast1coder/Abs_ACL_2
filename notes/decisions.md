# ABS decision log (date, decision, reason, who)

- 2026-11-11: Project set up. G0 = OLMo-2-1124-7B base vs instruct on PopQA with exact OLMo-2 counts.
  Primary variant S1_plain (shipped behaviour); S2_abstain is the prompted-abstention variant. (PI + Claude)
- 2026-11-11: Thresholds pre-registered in configs/g0.yaml:
  - GO if TT(S1) ≥ 1.5 with CI lower bound > 1.0;
  - FOLD if TT < 1.2 for all variants or tail base accuracy < 0.15;
  - tail = buckets {1, 2}, each needing ≥ 400 facts. (PI + Claude)
- 2026-11-11: Correctness = PopQA substring metric. Abstention = regex with no correct answer present. A hedged
  correct answer counts as correct. The regex must be hand-validated on 200 responses before G0 is signed. (Claude)
- 2026-11-11: Base confidence = agreement of 10 pure samples (T = 1, top_p = 1, top_k = 0) with the greedy answer.
  This feeds the matched-coverage baseline. (Claude)
- 2026-11-11: Area moved to Special Theme (knowledge collapse caused by safety training). (PI)
- 2026-11-11: Answer-alias filter (before any G0 data). PopQA alias lists contain "can"/"ca" (Canada), "in" (India),
  "US" (normalises to "us") and emoji flags. With raw lists, whole-word grading marked "It is not clear to us." correct
  for 310 facts and "I can't answer that." correct for 45 (an abstention scored as correct), and counts were inflated.
  Rule: keep PopQA `obj` first; keep another alias only if it normalises to >= 4 characters and is not an English
  stopword (sklearn list). Raw list kept as `answers_raw`. After the filter the probes are correct for 0 facts.
  Paper line: "We drop PopQA answer aliases shorter than four characters or that are stopwords; the canonical answer is
  always kept." (Claude)
- 2026-11-11: max_answers 8 -> 4. The infini-gram API rejects more than 4 terms per OR clause, so every fact with more
  than 4 answers failed (first counts run). Queries use the canonical answer plus up to 3 filtered aliases. Validation
  errors are no longer retried. (Claude)
- 2026-11-11: OPEN, needs PI. Approximate counts. When an answer term exceeds max_clause_freq (500k, the API maximum),
  infini-gram estimates the CNF count from a sample (approx=true). Example: "George Rankin AND politician" gives 0
  (approx), with "politician" at about 34M occurrences, so roughly 1.5% are sampled. Estimates are 0 or large, almost
  never 1 or 2, so facts with common answers (occupation, country, genre, sport) cannot enter the tail. Size of the
  effect: see the P0 report. (Claude)
