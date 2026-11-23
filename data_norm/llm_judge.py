"""LLM soft normalization: decide if a free-text correction matches a y16 category.

Uses Azure OpenAI GPT-4o via tokens.py (same pattern as counteredit/rankedit).
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import AzureOpenAI

import tokens  # noqa: E402  — local tokens.py with azure_key / azure_endpoint

AZURE_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT = "gpt-4o"

SYSTEM = (
    "You are a board-certified dermatologist and dermatopathologist. "
    "You will be given a clinician's free-text diagnosis and a pathology-confirmed "
    "diagnostic category from a 16-class taxonomy. Decide whether the free-text "
    "diagnosis is a clinically reasonable description of the pathology category — "
    "i.e. the same disease or a recognized clinical synonym/subtype that falls "
    "under that category.\n\n"
    "Respond with a JSON object: {\"match\": true/false, \"reason\": \"<one sentence>\"}."
)

PROMPT = (
    "Free-text diagnosis: \"{human_raw}\"\n"
    "Pathology category: \"{gt_y16}\"\n\n"
    "Does the free-text diagnosis clinically match this pathology category?"
)

_CLIENT: AzureOpenAI | None = None


def _client() -> AzureOpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AzureOpenAI(
            api_key=tokens.azure_key,
            api_version=AZURE_API_VERSION,
            azure_endpoint=tokens.azure_endpoint.rsplit("/openai/", 1)[0],
        )
    return _CLIENT


def judge(human_raw: str, gt_y16: str) -> dict:
    """Return {"match": bool, "reason": str}."""
    resp = _client().chat.completions.create(
        model=AZURE_DEPLOYMENT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(human_raw=human_raw, gt_y16=gt_y16)},
        ],
    )
    return json.loads(resp.choices[0].message.content)
