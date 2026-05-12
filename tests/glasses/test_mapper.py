import pytest

from glasses.mapper import SemanticMapper


class DummyModel:
    pass


def _build_mapper(process_token_batch):
    mapper = SemanticMapper.__new__(SemanticMapper)
    mapper.model = DummyModel()
    mapper.token_candidates_cache = {}
    mapper._process_token_batch = process_token_batch
    return mapper


def test_generate_token_candidates_raises_on_batch_failure():
    def _fail(batch, context=None):
        raise RuntimeError("boom")

    mapper = _build_mapper(_fail)

    with pytest.raises(RuntimeError, match="Failed to generate token candidates"):
        mapper.generate_token_candidates(["query"], max_workers=1)


def test_generate_token_candidates_raises_when_terms_missing_from_batch_result():
    def _partial(batch, context=None):
        return {batch[0]: ["lookup"]}

    mapper = _build_mapper(_partial)

    with pytest.raises(RuntimeError, match="Missing token candidates"):
        mapper.generate_token_candidates(["query", "model"], max_workers=1)
