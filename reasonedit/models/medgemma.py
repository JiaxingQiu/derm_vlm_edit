"""MedGemma (google/medgemma-1.5-4b-it) support for the VQAModel wrapper.

Mirrors the llava/qwen3 pattern: a loader, a processor getter, and a
chat-template preprocess that returns a batched inputs dict on CPU (the wrapper
moves it to device).

Added for the derm ReasonEdit experiment; not part of upstream reasonedit.
"""

import os
import torch


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def get_hf_model_medgemma(config, cache_dir=None, torch_dtype=None):
    from transformers import AutoModelForImageTextToText

    model_id = getattr(config.model, "pt", None) or config.model.name
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if cache_dir is not None and not getattr(config.model, "pt", None):
        load_kwargs["cache_dir"] = cache_dir
    token = _hf_token()
    if token:
        load_kwargs["token"] = token
    return AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)


def get_processor_medgemma(config, cache_dir=None):
    from transformers import AutoProcessor

    model_id = getattr(config.model, "pt", None) or config.model.name
    kwargs = {"trust_remote_code": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = _hf_token()
    if token:
        kwargs["token"] = token
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def preprocess_medgemma(images, prompts, processor, tokenize=False):
    """Build batched chat-template inputs for MedGemma.

    Mirrors the canonical derm_vlms loader
    (collect_ai_response/medgemma/utils.py): keep the image *inside* the message
    content and let ``apply_chat_template`` tokenize in one step
    (``tokenize=True, return_dict=True``). Do NOT run a second ``processor()``
    pass over pre-templated text -- that double-expands Gemma3's image soft
    tokens, so the image-token count no longer matches the image features and the
    forward pass hits a device-side assert. The ``tokenize`` arg is ignored; we
    always return the full inputs dict (input_ids, attention_mask, pixel_values).
    """
    imgs = images if isinstance(images, list) else [images]
    prs = prompts if isinstance(prompts, list) else [prompts]

    def _msg(img, pr):
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": pr},
            ],
        }]

    # Single example: mirror the canonical derm_vlms loader EXACTLY (a single
    # conversation, no batched list, no padding). This is the known-good path.
    if len(imgs) == 1:
        return processor.apply_chat_template(
            _msg(imgs[0], prs[0]),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # Batched fallback (unused at batch_size=1). Left-pad for correct generation.
    convs = [_msg(img, pr) for img, pr in zip(imgs, prs)]
    prev_side = getattr(processor.tokenizer, "padding_side", "left")
    processor.tokenizer.padding_side = "left"
    try:
        return processor.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    finally:
        processor.tokenizer.padding_side = prev_side
