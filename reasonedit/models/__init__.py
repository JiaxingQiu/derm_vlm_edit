"""VLM loaders used by VQAModel: MedGemma and DermatoLlama only."""

import math
from typing import Dict

import torch

from .medgemma import (
    get_hf_model_medgemma,
    get_processor_medgemma,
    preprocess_medgemma,
)
from .dermatollama import (
    get_hf_model_dermatollama,
    get_processor_dermatollama,
    preprocess_dermatollama,
)


def _is_medgemma(name_lower: str) -> bool:
    return "gemma" in name_lower


def _is_dermatollama(name_lower: str) -> bool:
    return "dermato" in name_lower or "mllama" in name_lower or (
        "llama" in name_lower and "llava" not in name_lower
    )


def _name(config) -> str:
    return getattr(getattr(config, "model", {}), "name", "").lower()


def get_processor(config):
    name_lower = _name(config)
    if _is_medgemma(name_lower):
        return get_processor_medgemma(config)
    if _is_dermatollama(name_lower):
        return get_processor_dermatollama(config)
    raise ValueError(f"Unsupported model: {getattr(config.model, 'name', None)}")


def get_tokenizer(config):
    return get_processor(config).tokenizer


def get_hf_model(config):
    name_lower = _name(config)
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    if _is_medgemma(name_lower):
        return get_hf_model_medgemma(config, cache_dir=None, torch_dtype=torch_dtype)
    if _is_dermatollama(name_lower):
        return get_hf_model_dermatollama(config, cache_dir=None, torch_dtype=torch_dtype)
    raise ValueError(f"Unsupported model: {getattr(config.model, 'name', None)}")


def get_preprocess(config):
    name_lower = _name(config)
    if _is_medgemma(name_lower):
        return preprocess_medgemma
    if _is_dermatollama(name_lower):
        return preprocess_dermatollama
    raise ValueError(f"Unsupported model: {getattr(config.model, 'name', None)}")


def clean_answer(o, i):
    s = o or ""
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[-1].strip()
    elif "assistant" in s:
        s = s.split("assistant")[-1].replace("\n", "").replace(":", "").strip()
    elif "Answer:" in s:
        s = s.split("Answer:")[-1].strip()
    elif "ANSWER:" in s:
        s = s.split("ANSWER:")[-1].strip()
    elif "answer:" in s:
        s = s.split("answer:")[-1].strip()
    else:
        s = s.replace(i, "").strip()
    return s


def nll_to_probs(label_losses: Dict[str, Dict[str, float]], use_avg: bool = False, temperature: float = 1.0) -> Dict[str, float]:
    scores = {}
    temp = max(1e-8, float(temperature))
    for lbl, d in label_losses.items():
        nll = float(d["avg_nll"] if use_avg else d["sum_nll"])
        scores[lbl] = -(nll) / temp
    m = max(scores.values()) if scores else 0.0
    exps = {lbl: math.exp(s - m) for lbl, s in scores.items()}
    Z = sum(exps.values()) or 1.0
    return {lbl: v / Z for lbl, v in exps.items()}
