"""Print candidate vision/language MLP layer names for a model.

Use this to fill inner_params_vision / inner_params_lang in the model yaml
(config/model/<model>.yaml). Pick a LATE block's MLP weight per convention.

Example:
    python tools/inspect_layers.py --model dermatollama
    python tools/inspect_layers.py --model medgemma --grep mlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import config as C  # noqa: E402

sys.path.insert(0, str(C.DERM_EDIT_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODEL_SHORT_TO_ANNOT.keys()))
    ap.add_argument("--grep", default="mlp", help="substring filter (default: mlp)")
    args = ap.parse_args()

    import torch
    from reasonedit import VQAModel, load_config

    config = load_config(
        args.model,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    model = VQAModel(config)

    names = [n for n, _ in model.model.named_parameters()]
    print(f"\ntotal params: {len(names)}")
    print("=" * 30, "TOP-LEVEL CHILDREN of model.model")
    for cn, _ in model.model.named_children():
        print("  ", cn)

    for kind, keys in [("VISION", ["vision", "visual"]), ("LANGUAGE", ["language", "text"])]:
        print("=" * 30, f"{kind} candidates (contains '{args.grep}', ends with .weight)")
        cand = [
            n for n in names
            if args.grep in n.lower() and any(k in n.lower() for k in keys) and n.endswith("weight")
        ]
        for n in cand:
            print("  ", n)


if __name__ == "__main__":
    main()
