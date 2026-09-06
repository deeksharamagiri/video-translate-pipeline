"""
Test evidence for the IndicTrans2 model-cache eviction bound
(INDICTRANS2_MAX_RESIDENT_MODELS, config.py).

pipeline/translate.py caches each loaded IndicTrans2 checkpoint (up to 3
distinct direction-specific models) in a module-level dict with no
eviction previously -- a long-running deployment serving more than one
direction over its lifetime would accumulate all of them resident at
once, a real risk on a memory-constrained (e.g. 8GB, no GPU) target.
These tests exercise the eviction logic directly with fake model/
tokenizer objects -- no real weights are loaded, so they run fast and
fully offline.
"""
import transformers
from IndicTransToolkit import processor as indictrans_processor_module

from pipeline import translate


class _FakeModel:
    def __init__(self, name):
        self.name = name

    def eval(self):
        return self


class _FakeTokenizer:
    def __init__(self, name):
        self.name = name


def _install_fakes(monkeypatch):
    load_calls = []

    def fake_model_from_pretrained(model_name, **kwargs):
        load_calls.append(model_name)
        return _FakeModel(model_name)

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        return _FakeTokenizer(model_name)

    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(fake_tokenizer_from_pretrained),
    )
    monkeypatch.setattr(
        indictrans_processor_module,
        "IndicProcessor",
        lambda inference=True: object(),
    )
    # Not testing quantization here -- avoid running real
    # torch.ao.quantization.quantize_dynamic against a fake model.
    monkeypatch.setattr(
        translate,
        "_maybe_quantize",
        lambda model: model,
    )

    return load_calls


def _reset_cache():
    translate._indictrans_cache.clear()
    translate._indictrans_processor.clear()


def test_cache_stays_within_configured_bound(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(translate, "INDICTRANS2_MAX_RESIDENT_MODELS", 1)
    _install_fakes(monkeypatch)

    translate._get_indictrans_model("model-a")
    assert list(translate._indictrans_cache.keys()) == ["model-a"]

    translate._get_indictrans_model("model-b")
    assert list(translate._indictrans_cache.keys()) == ["model-b"], (
        "loading a second checkpoint should have evicted the first "
        "to stay within INDICTRANS2_MAX_RESIDENT_MODELS=1"
    )


def test_cache_hit_does_not_reload_and_marks_recency(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(translate, "INDICTRANS2_MAX_RESIDENT_MODELS", 2)
    load_calls = _install_fakes(monkeypatch)

    translate._get_indictrans_model("model-a")
    translate._get_indictrans_model("model-b")
    assert load_calls == ["model-a", "model-b"]

    # Re-requesting model-a should be a cache hit (no new load call) and
    # should mark it as most-recently-used.
    translate._get_indictrans_model("model-a")
    assert load_calls == ["model-a", "model-b"], (
        "a cached model should not be reloaded from disk/network"
    )
    assert list(translate._indictrans_cache.keys()) == ["model-b", "model-a"]

    # Now loading a third, distinct model should evict model-b (the
    # least-recently-used) rather than model-a (just re-accessed).
    translate._get_indictrans_model("model-c")
    assert list(translate._indictrans_cache.keys()) == ["model-a", "model-c"], (
        "the least-recently-used entry should be evicted, not the one "
        "just accessed"
    )


def test_bound_is_never_exceeded_across_many_distinct_models(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(translate, "INDICTRANS2_MAX_RESIDENT_MODELS", 2)
    _install_fakes(monkeypatch)

    for name in ("model-a", "model-b", "model-c", "model-d", "model-e"):
        translate._get_indictrans_model(name)
        assert len(translate._indictrans_cache) <= 2, (
            "cache grew past INDICTRANS2_MAX_RESIDENT_MODELS while "
            f"loading {name!r}"
        )
