from src.agents.code_planner import _candidate_models as _karpathy_candidate_models
from src.agents.code_planner import _select_fallback_candidate
from src.agents.planner import _parse_config_from_llm, _random_fallback
from src.models import RetrievalConfig


def _base_config() -> RetrievalConfig:
    return RetrievalConfig(
        strategy="dense",
        embedding_model="BAAI/bge-base-en-v1.5",
        top_k=8,
        use_reranker=False,
    )


def test_parse_config_from_llm_clamps_and_applies_fields():
    payload = """{
      \"strategy\": \"hybrid\",
      \"top_k\": 999,
      \"use_reranker\": true,
      \"bm25_weight\": 2.0,
      \"dense_weight\": -1.0,
      \"query_rewrite\": true,
      \"source_diversity\": false,
      \"rationale\": \"try better balance\"
    }"""

    parsed = _parse_config_from_llm(payload, _base_config())
    assert parsed is not None
    config, rationale = parsed

    assert config.strategy == "hybrid"
    assert config.top_k == 20
    assert config.use_reranker is True
    assert config.reranker_model is not None
    assert config.bm25_weight == 1.0
    assert config.dense_weight == 0.0
    assert config.extra["query_rewrite"] is True
    assert config.extra["source_diversity"] is False
    assert rationale == "try better balance"


def test_random_fallback_respects_bounds():
    config, rationale = _random_fallback(_base_config(), iteration=1)

    assert config.strategy in {"dense", "hybrid"}
    assert 3 <= config.top_k <= 20
    assert 0.0 <= config.bm25_weight <= 1.0
    assert 0.0 <= config.dense_weight <= 1.0
    assert isinstance(rationale, str)
    assert len(rationale) > 0


def test_karpathy_code_planner_prefers_full_model_before_fast_model():
    models = _karpathy_candidate_models("gemini-pro", "gemini-flash")

    assert models[:2] == ["gemini-pro", "gemini-flash"]


def test_karpathy_fallback_moves_config_when_code_repeats():
    base = _base_config()
    state = {
        "current_pipeline_code": "def retrieve():\n    pass\n",
        "code_history": [],
    }

    code, config, rationale = _select_fallback_candidate(
        state,
        base,
        state["current_pipeline_code"],
    )

    assert code.strip() != state["current_pipeline_code"].strip()
    assert config.top_k != base.top_k or config.use_reranker != base.use_reranker
    assert rationale
