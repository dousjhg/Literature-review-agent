# Pre-existing components

Required in the submission: declare anything built before 22 October 2026.

| Component | What it is | Where it's used | Built by |
| --- | --- | --- | --- |
| Meterless World Model spec, TypeScript reference implementation and conformance suite ([Meterless/Meterless `engines/world-model`](https://github.com/Meterless/Meterless/tree/main/engines/world-model), Apache-2.0) | The design we implemented; our Python code in `litagent/integrations/world_model/` is ported from the reference, and our conformance tests port its suite. Licence copy: `litagent/integrations/world_model/LICENSE-meterless-world-model` | `litagent/integrations/world_model/`, `tests/test_world_model_conformance.py` | Meterless |
| Zetaris getting-started guides ([zetaris/getting-started](https://github.com/zetaris/getting-started)) | Connection settings and the REST SQL call our client uses | `litagent/integrations/zetaris.py`, `docs/setup-zetaris.md` | Zetaris |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0 | Pretrained semantic embeddings; pinned revision, prepared once and loaded locally on CPU | `litagent/integrations/embeddings.py`, Clusterer | Sentence Transformers |
| [SemCSE cosine](https://huggingface.co/CLAUSE-Bielefeld/SemCSE_cosine), MIT | Scientific-content encoder; documented CLS pooling; pinned revision, CPU-only offline loading | Default Clusterer representation | Marc Brinner, Sina Zarrieß / Bielefeld University |
| TODO | | | |

Open-source libraries listed in `requirements.txt` are third-party dependencies.

Everything else in this repository was built during the hackathon (22–27 October 2026).
