"""Prompts. B = base model (few-shot), S1 = instruct as shipped, S2 = instruct told to abstain."""
from __future__ import annotations

FEWSHOT = [
    ("What is the chemical symbol for gold?", "Au"),
    ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
    ("In what country is the city of Osaka?", "Japan"),
    ("What instrument did Louis Armstrong play?", "trumpet"),
    ("What is the capital city of Peru?", "Lima"),
]
FEWSHOT_SUBJECTS = {"gold", "Pride and Prejudice", "Osaka", "Louis Armstrong", "Peru"}


def base_prompt(question: str) -> str:
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FEWSHOT)
    return f"{shots}Q: {question}\nA:"


CHAT_VARIANTS = {
    "S1_plain": "{question} Answer briefly.",
    "S2_abstain": "{question} Answer briefly. If you are not sure of the answer, say \"I don't know\".",
}
