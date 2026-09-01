"""DermatoLlama support for the VQAModel wrapper.

DermatoLlama = meta-llama/Llama-3.2-11B-Vision-Instruct (Mllama) + a LoRA adapter
(DermaVLM/DermatoLLama-full). We mirror the *exact* working loader from the
derm_vlms repo (collect_ai_response/dermato_llama/utils.py): load the base model,
apply the adapter with PeftModel, and keep the live PeftModel wrapper (NO
merge_and_unload). Because the wrapper prefixes parameter names with
base_model.model.*, ReasonEdit's layer names are resolved at runtime via
validate_and_correct_param_name (see reliability/run_efficacy.resolve_layers).

Added for the derm ReasonEdit experiment; not part of upstream reasonedit.
"""

import os
import torch


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def get_hf_model_dermatollama(config, cache_dir=None, torch_dtype=None):
    from transformers import MllamaForConditionalGeneration
    from peft import PeftModel

    base_name = getattr(config.model, "pt", None) or config.model.name
    adapter_path = getattr(config.model, "adapter", None)
    token = _hf_token()

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if cache_dir is not None and not getattr(config.model, "pt", None):
        load_kwargs["cache_dir"] = cache_dir
    if token:
        load_kwargs["token"] = token

    model = MllamaForConditionalGeneration.from_pretrained(base_name, **load_kwargs)

    if adapter_path:
        peft_kwargs = {}
        if token:
            peft_kwargs["token"] = token
        # Keep the live PeftModel wrapper (matches the derm_vlms loader). Layer
        # names for ReasonEdit are resolved at runtime, so no merge is needed.
        model = PeftModel.from_pretrained(model, adapter_path, **peft_kwargs)

    model.eval()
    return model


def get_processor_dermatollama(config, cache_dir=None):
    from transformers import AutoProcessor

    # Processor comes from the BASE model (adapter has no processor).
    base_name = getattr(config.model, "pt", None) or config.model.name
    kwargs = {"trust_remote_code": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = _hf_token()
    if token:
        kwargs["token"] = token
    return AutoProcessor.from_pretrained(base_name, **kwargs)


def preprocess_dermatollama(images, prompts, processor, tokenize=False):
    """Build batched chat-template inputs for Mllama/DermatoLlama.

    Mllama's chat template inserts the <|image|> token itself and already adds
    BOS, so images are referenced by {"type": "image"} (no image object) and we
    pass add_special_tokens=False.
    """
    imgs = images if isinstance(images, list) else [images]
    prs = prompts if isinstance(prompts, list) else [prompts]
    convs = [[{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": pr},
        ],
    }] for pr in prs]
    texts = processor.apply_chat_template(
        convs,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Mllama requires images as a nested list (one sub-list of images per batch
    # element), matching the single <|image|> token in each templated text.
    nested_imgs = [[img] for img in imgs]
    return processor(
        text=texts,
        images=nested_imgs,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
