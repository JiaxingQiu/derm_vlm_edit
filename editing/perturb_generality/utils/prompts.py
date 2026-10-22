"""Rephrased question sentences. The numbered format suffix stays fixed."""

from __future__ import annotations

FORMAT = (
    "in format:\n"
    "1. [Diagnosis]: [Reasoning]\n"
    "2. [Diagnosis]: [Reasoning]\n"
    "3. [Diagnosis]: [Reasoning]"
)

SENTENCES = [
    "As a board-certified dermatologist, what are the three most likely diagnoses for this skin finding",
    "Acting as a dermatology specialist, what are your top three differential diagnoses",
    "You are a skilled dermatologist examining this image. What three diagnoses lead your differential",
    "From the perspective of an experienced dermatologist, which three conditions are most likely",
    "Imagine you are a consulting dermatologist. What are your three leading diagnostic possibilities",
    "Given your expertise in dermatology, which three diagnoses would you rank as most likely",
    "As an attending dermatologist, what three diagnoses would you consider first",
    "You are a dermatologist evaluating this clinical image. What is your top-three differential",
    "Please assume the role of an expert in dermatology. Which three diagnoses are most plausible",
    "In your capacity as a dermatology expert, what three diagnoses would you consider most likely",
]


def make_question(sentence: str) -> str:
    return f"{sentence.strip()}, {FORMAT}"
