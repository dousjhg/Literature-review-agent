"""Provision the pinned CPU encoder once (also called at Docker build time).

    python scripts/setup_embeddings.py
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litagent.config import (EMBEDDING_DOWNLOAD_TIMEOUT_S, EMBEDDING_METADATA_TIMEOUT_S,
                             EMBEDDING_PROFILES, Settings)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=EMBEDDING_PROFILES, default=None)
    args = parser.parse_args()
    # Set these before importing the Hub client, which reads them at import time.
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / "data/cache/huggingface"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(EMBEDDING_DOWNLOAD_TIMEOUT_S))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(EMBEDDING_METADATA_TIMEOUT_S))
    from huggingface_hub import snapshot_download
    from litagent.integrations.embeddings import embed_documents

    settings = Settings()
    if args.profile:
        settings.embedding_profile = args.profile
    profile = EMBEDDING_PROFILES[settings.embedding_profile]
    destination = Path(settings.embedding_directory)
    source = destination if profile["pooling"] == "mean" else root / f"data/cache/model-sources/{settings.embedding_profile}"
    snapshot_download(
        profile["model"], revision=profile["revision"], token=False,
        local_dir=str(source),
        allow_patterns=["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
                        "config.json", "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt", "model.safetensors", "README.md",
                        "spm.model", "added_tokens.json"],
    )
    if profile["pooling"] == "cls":
        # SemCSE's documented representation is the CLS token, not mean token pooling.
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
        encoder = Transformer(str(source), model_kwargs={"local_files_only": True, "trust_remote_code": False},
                              processor_kwargs={"local_files_only": True, "trust_remote_code": False},
                              config_kwargs={"local_files_only": True, "trust_remote_code": False})
        pooling = Pooling(encoder.get_embedding_dimension(), pooling_mode="cls")
        model = SentenceTransformer(modules=[encoder, pooling], device="cpu")
        model.save_pretrained(str(destination), create_model_card=False)
    (destination / "litagent-model.json").write_text(json.dumps(profile, indent=2) + "\n")
    vectors = embed_documents(["Literature review of language models."], settings)
    print(f"Prepared {profile['model']}@{profile['revision']}")
    print(f"Local model: {settings.embedding_directory}; CPU smoke check: {vectors.shape}")


if __name__ == "__main__":
    main()
