# Phases and gates (binding; thresholds pre-registered 2026-11-11 in configs/g0.yaml)

## P0: laptop

Pass all of the following before moving to G0:
- tests pass;
- PopQA is loaded;
- counts exist for at least 95% of facts;
- buckets "1" and "2" have at least 400 facts each (if not, run E1 first).

## P1: G0

**Validity checks (all must pass):**
- V1: tail buckets have ≥ 400 facts each.
- V2: base accuracy in each tail bucket is ≥ 0.15.
- V3: there are ≥ 30 tail losses.
- V4: counts are present.
- The regex is hand-validated on 200 responses (recorded in decisions.md).

**Verdicts:**

| Verdict | Condition | Action |
|---|---|---|
| GO | TT(S1) ≥ 1.5 with CI lower bound > 1.0 | E1, E2 |
| REVIEW | between thresholds, or only S2 passes | PI decides and logs |
| FOLD | TT < 1.2 for every variant, or V2 fails | fold into MB5 as a section; write a postmortem |
| BLOCKED | counts missing or V1 fails | fix the data (E1) |

## P2: G1 (after E2)

- **Pass:** TT ≥ 1.5 for at least 2 of the 4 methods.
- **Otherwise:** narrow the claim to the methods where it holds.

## P3: G2 (after E5)

- **Pass:** the fix recovers at least 30% of tail loss at no more than +1 point of hallucination.
- **Otherwise:** drop the fix and keep the audit.

## P4: writing

- Freeze on 20 December.
- Red-team the draft against `docs/05`.
