"""Edit-set records -> ReasonEdit `ex` schema, plus generate loop.

Prompt is the interface question as-is (top-3 differential + reasoning).
After generate, `pred.answer` is the extracted top-1 name; `pred.raw` is
the full generation. Scoring uses top-1 only.
"""

from __future__ import annotations

import json
from typing import Dict, List

from PIL import Image

from data_engineering.extract_dx import extract_top1


class DermEditDataset:
    def __init__(self, config, edits: List[Dict]):
        self.config = config
        self.data = [self._to_ex(e, i) for i, e in enumerate(edits)]

    @staticmethod
    def _to_ex(e: Dict, i: int) -> Dict:
        rationale = (e.get("rationale") or "").strip()
        return {
            "uid": e["uid"],
            "idx": i,
            "image": e["image"],
            "question": e["question"],
            "answer": e["answer"],
            "rationale": rationale,
            "cot": (e.get("cot") or rationale),
            "model_top1": e.get("model_top1"),
            "human_top1_raw": e.get("human_top1_raw"),
            "human_top1_y16": e.get("human_top1_y16"),
            "case_id": e.get("case_id"),
            "login_id": e.get("login_id"),
        }

    def set_dataloader(self):
        """Stamp gold + prompt on each example (no DataLoader)."""
        for i, ex in enumerate(self.data):
            ex["idx"] = i
            ex["gold"] = {"label": str(ex["answer"]).lower().strip()}
            prompt = str(ex["question"]).strip()
            ex["prompt"] = prompt
            ex["prompt_orig"] = prompt

    def task_generate(self, model, use_cache=True, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = getattr(self.config, "max_new_tokens", 512)
        try:
            core = getattr(getattr(model, "model", model), "model", getattr(model, "model", model))
            cfg = getattr(core, "config", None)
            if cfg is not None:
                cfg.use_cache = use_cache
            if hasattr(core, "gradient_checkpointing_disable"):
                core.gradient_checkpointing_disable()
        except Exception:
            pass
        for ex in self.data:
            img = ex["image"]
            img = img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            outs = model.generate(
                [img], [ex["prompt"]],
                max_new_tokens=max_new_tokens, use_cache=use_cache,
            )
            raw = outs[0]
            top1 = extract_top1(raw)
            ex["pred"] = {
                "answer": top1,
                "label_text": top1.lower().strip(),
                "raw": raw,
            }


def load_edits(path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
