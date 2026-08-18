"""
Translation routing:

    BOTH INDIAN LANGUAGES -> IndicTrans2 indic-indic-1B (AI4Bharat)
    ONE SIDE IS ENGLISH   -> Krutrim-Translate (krutrim-ai-labs) — fast,
                             distilled specifically for this; IndicTrans2's
                             own en-indic-1B/indic-en-1B checkpoints are
                             ~114x slower on CPU for the same sentences
                             (measured), but stay available via the
                             "indictrans2-en" forced engine for comparison.

All branches get glossary injection before/after the model call.
"""
import json
import os
import re
from typing import List

from config import (
    INDIC_LANGS, INDICTRANS2_MODEL, INDICTRANS2_EN_INDIC_MODEL, INDICTRANS2_INDIC_EN_MODEL,
    GLOSSARY_PATH, TRANSLATION_DEVICE, resolve_torch_device,
    KRUTRIM_TRANSLATE_REPO, KRUTRIM_TRANSLATE_DIR, KRUTRIM_TRANSLATE_DEVICE,
)

_indictrans_cache = {}
_krutrim_cache = {}


def load_glossary():
    if not os.path.exists(GLOSSARY_PATH):
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_readme", None)
    return data


def glossary_version_tag() -> str:
    """Cheap content hash so TM cache invalidates when glossary changes."""
    import hashlib
    g = load_glossary()
    return hashlib.md5(json.dumps(g, sort_keys=True).encode()).hexdigest()[:10]


def is_indic_pair(source_lang: str, target_lang: str) -> bool:
    return source_lang in INDIC_LANGS and target_lang in INDIC_LANGS


def _protect_glossary_terms(text: str, target_lang: str, glossary: dict):
    """
    Replace glossary source terms with placeholder tokens before translation
    so the NMT model can't mangle them, then restore the forced translation
    afterwards. Returns (protected_text, restore_map).
    """
    restore_map = {}
    protected = text
    for i, (term, translations) in enumerate(glossary.items()):
        if target_lang not in translations:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        if pattern.search(protected):
            placeholder = f"§§GLOSS{i}§§"
            protected = pattern.sub(placeholder, protected)
            restore_map[placeholder] = translations[target_lang]
    return protected, restore_map


def _restore_glossary_terms(text: str, restore_map: dict) -> str:
    for placeholder, forced_translation in restore_map.items():
        text = text.replace(placeholder, forced_translation)
    return text


# ---------------------------------------------------------------- IndicTrans2
# All three IndicTrans2 checkpoints (indic-indic, en-indic, indic-en) share
# the same transformers + IndicProcessor loading/generation shape, so that
# part is factored out; each direction just gets its own cached checkpoint.
def _load_indictrans_checkpoint(model_name: str) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {model_name}. This model is 'gated' on Hugging Face: "
            "free to use, but you must (1) accept the terms once at "
            f"https://huggingface.co/{model_name} while logged in, and "
            "(2) run 'huggingface-cli login' locally (or set HF_TOKEN) before "
            f"first use. See README.md for exact steps. Original error: {e}"
        ) from e
    device = resolve_torch_device(TRANSLATION_DEVICE)
    model = model.to(device)
    model.eval()
    return {
        "model": model, "tokenizer": tokenizer, "device": device,
        "processor": IndicProcessor(inference=True), "torch": torch,
    }


def _get_indictrans_checkpoint(cache_key: str, model_name: str) -> dict:
    if cache_key not in _indictrans_cache:
        _indictrans_cache[cache_key] = _load_indictrans_checkpoint(model_name)
    return _indictrans_cache[cache_key]


def _generate_indictrans(checkpoint: dict, texts: List[str], src_code: str, tgt_code: str) -> List[str]:
    model, tokenizer = checkpoint["model"], checkpoint["tokenizer"]
    processor, torch, device = checkpoint["processor"], checkpoint["torch"], checkpoint["device"]

    batch = processor.preprocess_batch(texts, src_lang=src_code, tgt_lang=tgt_code)
    inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=tgt_code)


_INDIC_FLORES_MAP = {
    "hin": "hin_Deva", "mar": "mar_Deva", "ben": "ben_Beng", "tam": "tam_Taml",
    "tel": "tel_Telu", "kan": "kan_Knda", "mal": "mal_Mlym", "guj": "guj_Gujr",
    "pan": "pan_Guru", "urd": "urd_Arab", "ori": "ory_Orya", "asm": "asm_Beng",
    "nep": "npi_Deva", "san": "san_Deva", "mai": "mai_Deva", "eng": "eng_Latn",
}


def translate_with_indictrans2(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    checkpoint = _get_indictrans_checkpoint("indic_indic", INDICTRANS2_MODEL)
    src_code = _INDIC_FLORES_MAP.get(source_lang, "hin_Deva")
    tgt_code = _INDIC_FLORES_MAP.get(target_lang, "eng_Latn")
    return _generate_indictrans(checkpoint, texts, src_code, tgt_code)


def translate_with_indictrans2_en(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """
    Uses IndicTrans2's dedicated en-indic / indic-en 1B checkpoints — the
    full-size (non-distilled) models Krutrim-Translate was distilled from,
    for comparison on English<->Indic naturalness. Much slower on CPU than
    Krutrim (measured ~114x) — not used by "auto" routing.
    """
    direction = "en_indic" if source_lang == "eng" else "indic_en"
    model_name = INDICTRANS2_EN_INDIC_MODEL if direction == "en_indic" else INDICTRANS2_INDIC_EN_MODEL
    checkpoint = _get_indictrans_checkpoint(direction, model_name)
    src_code = _INDIC_FLORES_MAP.get(source_lang, "eng_Latn")
    tgt_code = _INDIC_FLORES_MAP.get(target_lang, "eng_Latn")
    return _generate_indictrans(checkpoint, texts, src_code, tgt_code)


# --------------------------------------------------------------- Krutrim-Translate
def _get_krutrim_model(direction: str):
    """direction: 'en_indic' or 'indic_en' — Krutrim ships two directional checkpoints."""
    if direction not in _krutrim_cache:
        if not os.path.isdir(KRUTRIM_TRANSLATE_DIR):
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=KRUTRIM_TRANSLATE_REPO, local_dir=KRUTRIM_TRANSLATE_DIR)
            except Exception as e:
                raise RuntimeError(
                    "Could not download Krutrim-Translate. This model is 'gated' on "
                    "Hugging Face: free to use, but you must (1) accept the terms once "
                    f"at https://huggingface.co/{KRUTRIM_TRANSLATE_REPO} while logged in, "
                    "and (2) set the HF_TOKEN environment variable before first use. "
                    f"See README.md for exact steps. Original error: {e}"
                ) from e

        from pipeline.krutrim_engine.engine import Model as KrutrimModel
        ckpt_subdir = "ct_model_english_indic" if direction == "en_indic" else "ct_model_indic_english"
        ckpt_dir = os.path.join(KRUTRIM_TRANSLATE_DIR, ckpt_subdir)
        _krutrim_cache[direction] = KrutrimModel(
            ckpt_dir, device=KRUTRIM_TRANSLATE_DEVICE, input_lang_code_format="flores"
        )
    return _krutrim_cache[direction]


def translate_with_krutrim(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    direction = "en_indic" if source_lang == "eng" else "indic_en"
    model = _get_krutrim_model(direction)
    src_code = _INDIC_FLORES_MAP.get(source_lang, "eng_Latn")
    tgt_code = _INDIC_FLORES_MAP.get(target_lang, "eng_Latn")
    return model.batch_translate(texts, src_lang=src_code, tgt_lang=tgt_code, beam_len=5)


# --------------------------------------------------------------- entry point
def translate_batch(texts: List[str], source_lang: str, target_lang: str,
                     engine_override: str = "auto") -> tuple:
    """
    Route to the indic-indic or en-indic/indic-en IndicTrans2 checkpoint,
    with glossary protection wrapped around either.

    engine_override:
      "auto"           -> both Indic -> indic-indic-1B, else (English side) -> Krutrim-Translate
      "indictrans2"    -> force indic-indic-1B regardless of language pair (useful for
                          comparison; quality isn't guaranteed outside Indic<->Indic)
      "indictrans2-en" -> force IndicTrans2's en-indic-1B/indic-en-1B checkpoints (the
                          full-size models Krutrim was distilled from) — much slower on
                          CPU (~114x measured), useful only for A/B comparison
      "krutrim"        -> force Krutrim-Translate regardless of language pair (only
                          actually supports English<->one of the 9 KRUTRIM_INDIC_LANGS)
    """
    if not texts:
        return [], "n/a"

    glossary = load_glossary()
    protected_texts = []
    restore_maps = []
    for t in texts:
        protected, restore_map = _protect_glossary_terms(t, target_lang, glossary)
        protected_texts.append(protected)
        restore_maps.append(restore_map)

    use_indictrans2 = (
        engine_override == "indictrans2"
        or (engine_override == "auto" and is_indic_pair(source_lang, target_lang))
    )
    use_indictrans2_en = not use_indictrans2 and engine_override == "indictrans2-en"

    if use_indictrans2:
        engine_name = "IndicTrans2 (forced)" if engine_override == "indictrans2" else "IndicTrans2"
        raw_outputs = translate_with_indictrans2(protected_texts, source_lang, target_lang)
    elif use_indictrans2_en:
        engine_name = "IndicTrans2 en-indic-1B (forced)"
        raw_outputs = translate_with_indictrans2_en(protected_texts, source_lang, target_lang)
    else:
        engine_name = "Krutrim-Translate (forced)" if engine_override == "krutrim" else "Krutrim-Translate"
        raw_outputs = translate_with_krutrim(protected_texts, source_lang, target_lang)

    final_outputs = [
        _restore_glossary_terms(out, rmap)
        for out, rmap in zip(raw_outputs, restore_maps)
    ]
    return final_outputs, engine_name


def model_version_tag(source_lang: str, target_lang: str, engine_override: str = "auto") -> str:
    if engine_override == "indictrans2":
        return "indictrans2-1B-forced"
    if engine_override == "indictrans2-en":
        return "indictrans2-en-indic-1B-forced"
    if engine_override == "krutrim":
        return "krutrim-translate-forced"
    if is_indic_pair(source_lang, target_lang):
        return "indictrans2-1B"
    return "krutrim-translate"
