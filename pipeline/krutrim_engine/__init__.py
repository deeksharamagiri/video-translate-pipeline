# Vendored CTranslate2 inference wrapper for krutrim-ai-labs/Krutrim-Translate,
# sourced from https://github.com/ola-krutrim/KrutrimTranslate (the model's
# official reference implementation — the HF repo ships CTranslate2-exported
# weights only, no transformers/AutoModel-compatible checkpoint, so this glue
# code is required to run it. Trimmed to the ctranslate2 + batch_translate
# path this project uses; see engine.py for the two local modifications
# (lazy mosestokenizer/nltk imports, configurable compute_type).
