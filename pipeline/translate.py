"""
Translation routing exactly as drawn on the diagram:

    BOTH INDIAN LANGUAGES  -> IndicTrans2 (AI4Bharat)
    ANY OTHER LANGUAGE PAIR -> NLLB-200-distilled-600M (Meta AI)

Both branches get glossary injection before/after the model call.
"""
import json
import os
import re
from typing import Callable, List, Optional

from config import (
    INDIC_LANGS, INDICTRANS2_MODELS, INDICTRANS2_MODEL_SIZE, NLLB_MODEL,
    GLOSSARY_PATH, QUANTIZE_TRANSLATION_MODELS,
)

_indictrans_cache = {}      # model_name -> {"model":, "tokenizer":}
_indictrans_processor = {}  # lazy-init once, shared across checkpoints
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


def indictrans2_direction(source_lang: str, target_lang: str) -> Optional[str]:
    """Return 'en_indic' | 'indic_en' | 'indic_indic' | None (not an IndicTrans2 pair).

    AI4Bharat ships three direction-specific checkpoints. Previously only the
    indic-indic checkpoint was wired up, so English<->Indic pairs silently
    fell through to NLLB even though a dedicated checkpoint exists for them.
    """
    src_indic = source_lang in INDIC_LANGS
    tgt_indic = target_lang in INDIC_LANGS
    if src_indic and tgt_indic:
        return "indic_indic"
    if source_lang == "eng" and tgt_indic:
        return "en_indic"
    if src_indic and target_lang == "eng":
        return "indic_en"
    return None


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


_quant_warned = False


def _maybe_quantize(model):
    """Dynamic INT8 quantization of nn.Linear layers -- CPU speedup for
    translation models. Weights-only, doesn't touch embeddings, safe under
    .generate()/beam search. Version-agnostic import since the exact
    namespace has moved across torch releases.

    torch.backends.quantized.engine defaults to 'none' on many platforms
    (notably macOS/arm64, where only 'qnnpack' is actually available) --
    quantize_dynamic raises "Didn't find engine for operation
    quantized::linear_prepack NoQEngine" if it's left unset. Pick a usable
    engine before quantizing; if none is available at all, skip
    quantization and run the model unquantized rather than crash the job.
    """
    global _quant_warned
    if not QUANTIZE_TRANSLATION_MODELS:
        return model
    import torch
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic

    if torch.backends.quantized.engine == "none":
        supported = torch.backends.quantized.supported_engines
        preferred = next((e for e in ("fbgemm", "x86", "qnnpack", "onednn") if e in supported), None)
        if preferred is None:
            if not _quant_warned:
                print("[translate] No PyTorch quantization engine available on this "
                      "platform -- running translation models unquantized (slower, "
                      "same output quality). Set QUANTIZE_TRANSLATION_MODELS=False "
                      "in config.py to silence this check.")
                _quant_warned = True
            return model
        torch.backends.quantized.engine = preferred

    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------- IndicTrans2
def _get_indictrans_model(model_name: str):
    if model_name not in _indictrans_cache:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not load IndicTrans2 checkpoint '{model_name}'. This model "
                "is 'gated' on Hugging Face: free to use, but you must (1) accept "
                f"the terms once at https://huggingface.co/{model_name} while "
                "logged in, and (2) run 'huggingface-cli login' locally (or set "
                "HF_TOKEN) before first use. See README.md for exact steps. "
                f"Original error: {e}"
            ) from e
        model.eval()
        model = _maybe_quantize(model)
        _indictrans_cache[model_name] = {"model": model, "tokenizer": tokenizer}
        if "processor" not in _indictrans_processor:
            _indictrans_processor["processor"] = IndicProcessor(inference=True)
            _indictrans_processor["torch"] = torch
    entry = _indictrans_cache[model_name]
    return (entry["model"], entry["tokenizer"],
            _indictrans_processor["processor"], _indictrans_processor["torch"])


_INDIC_FLORES_MAP = {
    "hin": "hin_Deva", "mar": "mar_Deva", "ben": "ben_Beng", "tam": "tam_Taml",
    "tel": "tel_Telu", "kan": "kan_Knda", "mal": "mal_Mlym", "guj": "guj_Gujr",
    "pan": "pan_Guru", "urd": "urd_Arab", "ori": "ory_Orya", "asm": "asm_Beng",
    "nep": "npi_Deva", "san": "san_Deva", "eng": "eng_Latn",
}


def translate_with_indictrans2(texts: List[str], source_lang: str, target_lang: str,
                                direction: Optional[str] = None) -> List[str]:
    direction = direction or indictrans2_direction(source_lang, target_lang) or "indic_indic"
    model_name = INDICTRANS2_MODELS[direction][INDICTRANS2_MODEL_SIZE]
    model, tokenizer, processor, torch = _get_indictrans_model(model_name)
    src_code = _INDIC_FLORES_MAP.get(source_lang, "hin_Deva")
    tgt_code = _INDIC_FLORES_MAP.get(target_lang, "eng_Latn")

    batch = processor.preprocess_batch(texts, src_lang=src_code, tgt_lang=tgt_code)
    inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5,
            no_repeat_ngram_size=3, repetition_penalty=1.3,
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
        model = _maybe_quantize(model)
        _nllb_cache["tokenizer"] = tokenizer
        _nllb_cache["model"] = model
    return _nllb_cache["model"], _nllb_cache["tokenizer"]


def translate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """
    Batched NLLB translation. Previously this ran one text through
    tokenizer/model.generate() at a time in a Python loop even though the
    model was already loaded once -- a single padded batch of the whole
    chunk is meaningfully faster on CPU than N separate generate() calls,
    the same way translate_with_indictrans2 already batches its chunk.
    """
    import torch
    model, tokenizer = _get_nllb_model()
    src_code = NLLB_LANG_CODE_MAP.get(source_lang, "eng_Latn")
    tgt_code = NLLB_LANG_CODE_MAP.get(target_lang, "eng_Latn")

    tokenizer.src_lang = src_code
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        generated = model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id,
            max_length=256, num_beams=5,
            no_repeat_ngram_size=3, repetition_penalty=1.3,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# Sub-batch size for translate_batch's chunking loop. Keeps IndicTrans2's
# batched model.generate() call (much faster than one-text-at-a-time) while
# still giving the caller a progress_cb tick every few segments instead of
# one silent call covering the whole job.
TRANSLATE_CHUNK_SIZE = 8


# --------------------------------------------------------------- entry point
def translate_batch(texts: List[str], source_lang: str, target_lang: str,
                     engine_override: str = "auto",
                     progress_cb: Optional[Callable[[int, int], None]] = None) -> tuple:
    """
    Route to IndicTrans2 or NLLB-200 per the diagram, with glossary
    protection wrapped around either engine.

    engine_override:
      "auto"        -> both/either side Indic (incl. English via the dedicated
                       en-indic/indic-en checkpoints) -> IndicTrans2, else NLLB
      "indictrans2" -> force IndicTrans2 regardless of language pair (falls
                       back to the indic-indic checkpoint for pairs outside
                       its three trained directions)
      "nllb"        -> force NLLB-200-distilled-600M regardless of language pair

    progress_cb, if given, is called as progress_cb(done, total) after each
    sub-batch finishes, so callers can report live "X of Y segments" progress
    instead of one message that sits still for the whole translation step.
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

    direction = indictrans2_direction(source_lang, target_lang)
    use_indictrans2 = (
        engine_override == "indictrans2"
        or (engine_override == "auto" and direction is not None)
    )

    if use_indictrans2:
        resolved_direction = direction or "indic_indic"  # forced-but-out-of-scope-pair fallback
        model_name = INDICTRANS2_MODELS[resolved_direction][INDICTRANS2_MODEL_SIZE]
        short_name = model_name.split("/")[-1]
        engine_name = f"IndicTrans2 ({short_name})" + (" (forced)" if engine_override == "indictrans2" else "")
    else:
        engine_name = "NLLB-200-distilled-600M (forced)" if engine_override == "nllb" else "NLLB-200-distilled-600M"

    total = len(protected_texts)
    raw_outputs = []
    for start in range(0, total, TRANSLATE_CHUNK_SIZE):
        chunk = protected_texts[start:start + TRANSLATE_CHUNK_SIZE]
        if use_indictrans2:
            raw_outputs.extend(translate_with_indictrans2(chunk, source_lang, target_lang, direction))
        else:
            raw_outputs.extend(translate_with_nllb(chunk, source_lang, target_lang))
        if progress_cb:
            progress_cb(len(raw_outputs), total)

    final_outputs = [
        _restore_glossary_terms(out, rmap)
        for out, rmap in zip(raw_outputs, restore_maps)
    ]
    return final_outputs, engine_name


def model_version_tag(source_lang: str, target_lang: str, engine_override: str = "auto") -> str:
    quant_suffix = "-int8dyn" if QUANTIZE_TRANSLATION_MODELS else ""
    if engine_override == "nllb":
        return f"nllb-200-distilled-600M-forced{quant_suffix}"
    direction = indictrans2_direction(source_lang, target_lang)
    if engine_override == "indictrans2" or (engine_override == "auto" and direction is not None):
        resolved = direction or "indic_indic"
        short_name = INDICTRANS2_MODELS[resolved][INDICTRANS2_MODEL_SIZE].split("/")[-1]
        forced = "-forced" if engine_override == "indictrans2" else ""
        return f"{short_name}{forced}{quant_suffix}"
    return f"nllb-200-distilled-600M{quant_suffix}"