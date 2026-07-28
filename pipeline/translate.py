"""
Translation routing exactly as drawn on the diagram:

    BOTH INDIAN LANGUAGES  -> IndicTrans2 (AI4Bharat)
    ANY OTHER LANGUAGE PAIR -> NLLB-200-distilled-600M (Meta AI)

Both branches get glossary injection before/after the model call.
"""
import json
import os
import re
from typing import List

from config import INDIC_LANGS, INDICTRANS2_MODEL, NLLB_MODEL, GLOSSARY_PATH

_indictrans_cache = {}
_nllb_cache = {}

# NLLB uses FLORES-200 codes; map our internal ISO-639-3-ish codes to them.
NLLB_LANG_CODE_MAP = {
    "eng": "eng_Latn", "hin": "hin_Deva", "mar": "mar_Deva", "ben": "ben_Beng",
    "tam": "tam_Taml", "tel": "tel_Telu", "kan": "kan_Knda", "mal": "mal_Mlym",
    "guj": "guj_Gujr", "pan": "pan_Guru", "urd": "urd_Arab", "ori": "ory_Orya",
    "asm": "asm_Beng", "nep": "npi_Deva", "san": "san_Deva",
    "fra": "fra_Latn", "spa": "spa_Latn", "deu": "deu_Latn", "zho": "zho_Hans",
    "ara": "arb_Arab", "por": "por_Latn", "rus": "rus_Cyrl", "jpn": "jpn_Jpan",
}


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
def _get_indictrans_model():
    if "model" not in _indictrans_cache:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        try:
            tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_MODEL, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                INDICTRANS2_MODEL, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(
                "Could not load IndicTrans2. This model is 'gated' on Hugging Face: "
                "free to use, but you must (1) accept the terms once at "
                f"https://huggingface.co/{INDICTRANS2_MODEL} while logged in, and "
                "(2) run 'huggingface-cli login' locally (or set HF_TOKEN) before "
                f"first use. See README.md for exact steps. Original error: {e}"
            ) from e
        model.eval()
        _indictrans_cache["tokenizer"] = tokenizer
        _indictrans_cache["model"] = model
        _indictrans_cache["processor"] = IndicProcessor(inference=True)
        _indictrans_cache["torch"] = torch
    return (_indictrans_cache["model"], _indictrans_cache["tokenizer"],
            _indictrans_cache["processor"], _indictrans_cache["torch"])


_INDIC_FLORES_MAP = {
    "hin": "hin_Deva", "mar": "mar_Deva", "ben": "ben_Beng", "tam": "tam_Taml",
    "tel": "tel_Telu", "kan": "kan_Knda", "mal": "mal_Mlym", "guj": "guj_Gujr",
    "pan": "pan_Guru", "urd": "urd_Arab", "ori": "ory_Orya", "asm": "asm_Beng",
    "nep": "npi_Deva", "san": "san_Deva", "eng": "eng_Latn",
}


def translate_with_indictrans2(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    model, tokenizer, processor, torch = _get_indictrans_model()
    src_code = _INDIC_FLORES_MAP.get(source_lang, "hin_Deva")
    tgt_code = _INDIC_FLORES_MAP.get(target_lang, "eng_Latn")

    batch = processor.preprocess_batch(texts, src_lang=src_code, tgt_lang=tgt_code)
    inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=tgt_code)


# ---------------------------------------------------------------------- NLLB
def _get_nllb_model():
    if "model" not in _nllb_cache:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        model.eval()
        _nllb_cache["tokenizer"] = tokenizer
        _nllb_cache["model"] = model
    return _nllb_cache["model"], _nllb_cache["tokenizer"]


def translate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    import torch
    model, tokenizer = _get_nllb_model()
    src_code = NLLB_LANG_CODE_MAP.get(source_lang, "eng_Latn")
    tgt_code = NLLB_LANG_CODE_MAP.get(target_lang, "eng_Latn")

    tokenizer.src_lang = src_code
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            generated = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id,
                max_length=256, num_beams=5
            )
        results.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return results


# --------------------------------------------------------------- entry point
def translate_batch(texts: List[str], source_lang: str, target_lang: str,
                     engine_override: str = "auto") -> tuple:
    """
    Route to IndicTrans2 or NLLB-200 per the diagram, with glossary
    protection wrapped around either engine.

    engine_override:
      "auto"        -> diagram's routing logic (both Indic -> IndicTrans2, else NLLB)
      "indictrans2" -> force IndicTrans2 regardless of language pair (useful for
                       comparison; quality isn't guaranteed outside Indic<->Indic
                       since only the indic-indic checkpoint is wired up below)
      "nllb"        -> force NLLB-200-distilled-600M regardless of language pair
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

    if use_indictrans2:
        engine_name = "IndicTrans2 (forced)" if engine_override == "indictrans2" else "IndicTrans2"
        raw_outputs = translate_with_indictrans2(protected_texts, source_lang, target_lang)
    else:
        engine_name = "NLLB-200-distilled-600M (forced)" if engine_override == "nllb" else "NLLB-200-distilled-600M"
        raw_outputs = translate_with_nllb(protected_texts, source_lang, target_lang)

    final_outputs = [
        _restore_glossary_terms(out, rmap)
        for out, rmap in zip(raw_outputs, restore_maps)
    ]
    return final_outputs, engine_name


def model_version_tag(source_lang: str, target_lang: str, engine_override: str = "auto") -> str:
    if engine_override == "indictrans2":
        return "indictrans2-1B-forced"
    if engine_override == "nllb":
        return "nllb-200-distilled-600M-forced"
    return "indictrans2-1B" if is_indic_pair(source_lang, target_lang) else "nllb-200-distilled-600M"
