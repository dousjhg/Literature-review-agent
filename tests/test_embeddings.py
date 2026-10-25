"""Offline loading and token-window pooling must not silently discard long abstracts."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from litagent.config import Settings
from litagent.integrations import embeddings
from litagent.trace import Trace


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False, "verbose": False}
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class Encoder:
    tokenizer = Tokenizer()
    max_seq_length = 6

    def __init__(self):
        self.chunks = []

    def encode(self, chunks, **kwargs):
        self.chunks = chunks
        assert kwargs["normalize_embeddings"] and not kwargs["show_progress_bar"]
        return np.array([[1, 0] if chunk.startswith("a") else [0, 1] for chunk in chunks], dtype=float)


def test_long_documents_use_multiple_windows_and_normalized_pooling(monkeypatch):
    encoder = Encoder()
    monkeypatch.setattr(embeddings, "_encoder", lambda path: encoder)
    out = embeddings.embed_documents(["a b c d e f", "x y"], Settings())
    assert encoder.chunks == ["a b c d", "e f", "x y"]
    np.testing.assert_allclose(out, [[2 ** -0.5, 2 ** -0.5], [0, 1]], rtol=1e-6)


def test_chunk_cap_is_logged(monkeypatch, tmp_path):
    encoder = Encoder()
    monkeypatch.setattr(embeddings, "_encoder", lambda path: encoder)
    with Trace(str(tmp_path)) as trace:
        embeddings.embed_documents(["a b c d e f"], Settings(embedding_max_chunks_per_paper=1))
    assert encoder.chunks == ["a b c d"]
    assert "1 documents capped" in trace.events[-1]["why"]


def test_model_is_loaded_from_disk_with_network_and_remote_code_disabled(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text("[]")
    calls = []

    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return Encoder()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    embeddings._encoder.cache_clear()
    try:
        embeddings._encoder(str(tmp_path))
        embeddings._encoder(str(tmp_path))
    finally:
        embeddings._encoder.cache_clear()
    assert len(calls) == 1
    assert calls[0][1] == {"device": "cpu", "local_files_only": True, "token": False, "trust_remote_code": False}


def test_missing_weights_raise_actionable_error_without_downloading(tmp_path):
    with pytest.raises(embeddings.EmbeddingError, match="setup_embeddings.py"):
        embeddings._encoder(str(tmp_path))


def test_empty_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(embeddings, "_encoder", lambda path: pytest.fail("loaded model for empty input"))
    assert embeddings.embed_documents([], Settings()).shape == (0, 0)


def test_blank_document_is_rejected(monkeypatch):
    monkeypatch.setattr(embeddings, "_encoder", lambda path: Encoder())
    with pytest.raises(embeddings.EmbeddingError, match="empty paper"):
        embeddings.embed_documents([" "], Settings())


def test_profile_mismatch_is_rejected_instead_of_logging_a_false_model_name(monkeypatch):
    encoder = Encoder()
    encoder.litagent_metadata = {"model": "some-other-model", "revision": "wrong", "pooling": "mean"}
    monkeypatch.setattr(embeddings, "_encoder", lambda path: encoder)
    with pytest.raises(embeddings.EmbeddingError, match="does not match"):
        embeddings.embed_documents(["a b"], Settings())
