"""CPU document embeddings. Model provisioning is explicit; runtime never downloads files."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from litagent.config import EMBEDDING_PROFILES, Settings
from litagent.trace import current_trace


class EmbeddingError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _encoder(directory: str):
    path = Path(directory).expanduser().resolve()
    if not (path / "modules.json").is_file():
        raise EmbeddingError("Embedding model is missing. Run python scripts/setup_embeddings.py first "
                             "(or set EMBEDDING_MODEL_DIR to a prepared local model).")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EmbeddingError("Install requirements.txt before running real clustering.") from e
    try:
        encoder = SentenceTransformer(str(path), device="cpu", local_files_only=True, token=False,
                                      trust_remote_code=False)
        metadata = path / "litagent-model.json"
        encoder.litagent_metadata = json.loads(metadata.read_text()) if metadata.is_file() else None
        return encoder
    except (OSError, ValueError) as e:
        raise EmbeddingError("Cannot load the local embedding model; rerun scripts/setup_embeddings.py.") from e


def embed_documents(texts: list[str], settings: Settings) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    profile = EMBEDDING_PROFILES.get(settings.embedding_profile)
    if profile is None:
        raise EmbeddingError(f"Unknown embedding profile: {settings.embedding_profile}")
    requested_window = settings.embedding_chunk_tokens if settings.embedding_chunk_tokens is not None else profile["chunk_tokens"]
    if min(settings.embedding_batch_size, requested_window,
           settings.embedding_max_chunks_per_paper) < 1:
        raise EmbeddingError("Embedding batch, chunk and document limits must be positive.")
    encoder = _encoder(settings.embedding_directory)
    metadata = getattr(encoder, "litagent_metadata", None)
    if metadata and any(metadata.get(key) != profile[key] for key in ("model", "revision", "pooling")):
        raise EmbeddingError("The local model does not match EMBEDDING_PROFILE; prepare the selected profile "
                             "or correct EMBEDDING_MODEL_DIR.")
    tokenizer = encoder.tokenizer
    window = min(requested_window,
                 encoder.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False))
    if window < 1:
        raise EmbeddingError("The embedding model has no space for document tokens.")
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    truncated = 0
    for text in texts:
        if not text.strip():
            raise EmbeddingError("Cannot embed an empty paper text.")
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        limit = window * settings.embedding_max_chunks_per_paper
        truncated += int(len(tokens) > limit)
        start = len(chunks)
        for offset in range(0, min(len(tokens), limit), window):
            chunks.append(tokenizer.decode(tokens[offset:offset + window], skip_special_tokens=True))
        if start == len(chunks):
            raise EmbeddingError("The tokenizer produced no tokens for a paper.")
        spans.append((start, len(chunks)))
    encoded = np.asarray(encoder.encode(chunks, batch_size=min(settings.embedding_batch_size, profile["batch_size"]),
                                       show_progress_bar=False, convert_to_numpy=True,
                                       normalize_embeddings=True), dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(chunks) or not np.isfinite(encoded).all():
        raise EmbeddingError("The encoder returned invalid chunk embeddings.")
    documents = np.stack([encoded[start:end].mean(axis=0) for start, end in spans])
    norms = np.linalg.norm(documents, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise EmbeddingError("The encoder returned a zero document vector.")
    if trace := current_trace():
        trace.log("clusterer", "embed papers", f"{len(texts)} documents, {len(chunks)} chunks",
                  next_step="group by content similarity",
                  model=metadata["model"] if metadata else settings.embedding_directory,
                  revision=metadata["revision"] if metadata else None,
                  why=f"Local CPU encoder; normalized mean of token chunks; {truncated} documents capped at "
                      f"{settings.embedding_max_chunks_per_paper} chunks")
    return documents / norms
