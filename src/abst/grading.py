"""Grading: PopQA-style correctness and abstention detection.

correct(answer)  : any accepted answer string appears in the response (case-insensitive,
                   normalised whitespace and punctuation). This is the standard PopQA metric.
abstained(answer): the response declines (regex below) AND contains no accepted answer.
A hedged response that still contains a correct answer counts as correct, not as abstention.
The regex is validated by hand on 200 responses before G0 is signed (docs/03).
"""
from __future__ import annotations

import re
import unicodedata

ABSTAIN_RE = re.compile(
    r"(i don'?t know|i do not know|i'?m not sure|i am not sure|not sure|no information|"
    r"i (?:can ?not|can'?t|couldn'?t|could not) (?:find|answer|determine|provide|say|verify)|"
    r"(?:unable|not able) to (?:find|answer|determine|provide|verify)|i don'?t have (?:any |enough )?information|"
    r"there is no (?:widely )?(?:known|available|reliable) information|unknown|i'?m unable|"
    r"i do not have (?:any |enough )?information|not aware of)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def is_correct(response: str, answers: list[str]) -> bool:
    r = " " + norm(response) + " "
    for a in answers:
        na = norm(a)
        if na and (" " + na + " ") in r:
            return True
    return False


def is_abstention(response: str, answers: list[str]) -> bool:
    if is_correct(response, answers):
        return False
    if not norm(response):
        return True  # empty output = no answer given
    return bool(ABSTAIN_RE.search(response))


def same_answer(a: str, b: str) -> bool:
    """Agreement used for base self-consistency confidence."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb or (" " + na + " ") in (" " + nb + " ") or (" " + nb + " ") in (" " + na + " ")
